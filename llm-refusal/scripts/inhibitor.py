"""What is the refusal inhibitor? (PROPOSED_PLANS item 2)

The `--mode rank1` remove adapter (ΔW = u vᵀ on a down_proj just before r̂'s layer)
switches refusal off through the r̂-free part of u, yet ablating that direction from
the model leaves refusal intact. This asks how it works and what it is:

  gate        what the adapter responds to: v·x (its input projection) on harmful vs
              harmless prompts, at r̂'s position and the last token. The adapter writes
              u scaled by this gate, so u's sign alone means nothing; the *effective
              write* w = u · mean gate on the prompts it was trained to change.
  mechanism   with the adapter installed (full u, and u with r̂ removed), the residual
              stream's projection onto r̂ at r̂'s position, at every layer from the
              adapter on, on harmful prompts, beside the plain model and r̂ ablation.
              Does the inhibitor stop the refusal signal forming, or override it later?
  feature     in the plain model, the projection onto the inhibitor direction (w with
              r̂ removed, unit) at r̂'s layer and position: harmful vs harmless, AUROC.
  axis        cos between the remove and induce adapters' effective writes, and of
              each with r̂: is "install refusal" the same axis reversed?
  generality  with the inhibitor installed, empathy detection on empathy prompts.

    .venv/bin/python3 llm-refusal/scripts/inhibitor.py --model meta-llama/Meta-Llama-3-8B-Instruct

Writes results/finetune/<model>-inhibitor.json.
"""
import argparse
import contextlib
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
_high = os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(min(0.6, 0.75 * float(_high))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from concept import get_concept  # noqa: E402
from datatypes import DirectionVector  # noqa: E402
from evaluation import BigEvaluator  # noqa: E402
from finetune import adapted, adapter_sites  # noqa: E402
from formatting import last_real_token_indices  # noqa: E402
from framework import DirectionTestFramework  # noqa: E402

BATCH = 16


def auroc(pos, neg) -> float:
    """P(score of a random positive > a random negative); ties count half."""
    pos, neg = t.as_tensor(pos), t.as_tensor(neg)
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--remove-tag", default="remove-s0")
    ap.add_argument("--induce-tag", default="induce-s0")
    args = ap.parse_args()
    short = args.model.split("/")[-1]
    fw = DirectionTestFramework(model_name=args.model, concept="refusal")
    model, fmt, ev = fw.model, fw.prompt_formatter, fw.evaluator
    dev = model.device
    r = DirectionVector.load(f"results/{short}-refusal-direction")
    r_hat = r.unit.float().cpu()
    harmful, harmless = fw.concept.eval_data_fn()
    n_layers = len(fw.intervention_applier.transformer_layers)

    def load_adapter(tag):
        meta = json.load(open(f"results/finetune/{short}-refusal-rank1-{tag}.json"))
        w = t.load(f"results/finetune/{short}-refusal-rank1-{tag}-adapters.pt")[0]
        return meta["adapter_layers"][0], w["U"].float(), w["V"].float()
    layer, U_rem, V_rem = load_adapter(args.remove_tag)
    layer_ind, U_ind, V_ind = load_adapter(args.induce_tag)

    @contextlib.contextmanager
    def with_adapter(at_layer, U, V, gates=None):
        """Install a trained rank-1 adapter; optionally record its gate v·x per token."""
        with adapted(model, adapter_sites(model, [at_layer], ["down_proj"]), rank=1, seed=0) as (a,):
            a.U.data, a.V.data = U.to(dev), V.to(dev)
            hook = None
            if gates is not None:
                hook = a.register_forward_pre_hook(lambda mod, inp: gates.append((inp[0].float() @ mod.V)[..., 0].detach().cpu()))
            try:
                yield
            finally:
                if hook is not None:
                    hook.remove()

    @contextlib.contextmanager
    def nothing():
        yield

    def forward_stats(prompts, ctx, directions):
        """Per layer, the mean projection onto each direction at r̂'s position (and
        the per-prompt values at r̂'s layer), over `prompts` under `ctx`."""
        per_layer = {k: t.zeros(n_layers) for k in directions}
        at_r_layer = {k: [] for k in directions}
        with ctx():
            for i in range(0, len(prompts), BATCH):
                enc = fmt.format_batch(prompts[i:i + BATCH])
                with t.no_grad():
                    hs = model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                               position_ids=enc["position_ids"].to(dev), output_hidden_states=True).hidden_states
                idx = last_real_token_indices(enc["attention_mask"]) + 1 + r.position_index  # pos -1 = last real
                rows = t.arange(len(idx))
                for L in range(n_layers):                    # hidden_states[L] = input to layer L
                    h = hs[L][rows.to(dev), idx.to(dev)].float().cpu()
                    for k, d in directions.items():
                        proj = h @ d
                        per_layer[k][L] += proj.sum()
                        if L == r.layer:
                            at_r_layer[k] += proj.tolist()
        return {k: (v / len(prompts)).tolist() for k, v in per_layer.items()}, at_r_layer

    out = {"model": args.model, "remove_tag": args.remove_tag, "induce_tag": args.induce_tag,
           "adapter_layer": layer, "direction": {"layer": r.layer, "position_index": r.position_index}}

    # Gate: what the remove adapter responds to (and the induce adapter's, for the axis).
    def gate_stats(at_layer, U, V, prompts):
        gates = []
        with with_adapter(at_layer, U, V, gates):
            for i in range(0, len(prompts), BATCH):
                enc = fmt.format_batch(prompts[i:i + BATCH])
                with t.no_grad():
                    model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                          position_ids=enc["position_ids"].to(dev))
                g = gates.pop()
                last = last_real_token_indices(enc["attention_mask"])
                rows = t.arange(len(last))
                yield g[rows, last + 1 + r.position_index], g[rows, last], \
                    (g * enc["attention_mask"]).sum(-1) / enc["attention_mask"].sum(-1)
    def summarise(at_layer, U, V, prompts):
        at_r, at_last, mean_all = map(t.cat, zip(*gate_stats(at_layer, U, V, prompts)))
        return {"at_r_position": at_r.mean().item(), "at_last_token": at_last.mean().item(),
                "mean_over_prompt": mean_all.mean().item()}, at_r
    g_rem_h, g_rem_h_vals = summarise(layer, U_rem, V_rem, harmful)
    g_rem_b, g_rem_b_vals = summarise(layer, U_rem, V_rem, harmless)
    g_ind_h, _ = summarise(layer_ind, U_ind, V_ind, harmful)
    g_ind_b, _ = summarise(layer_ind, U_ind, V_ind, harmless)
    out["gate"] = {"remove": {"harmful": g_rem_h, "harmless": g_rem_b,
                              "auroc_harmful_vs_harmless": auroc(g_rem_h_vals.abs(), g_rem_b_vals.abs())},
                   "induce": {"harmful": g_ind_h, "harmless": g_ind_b}}
    print("gate v·x  remove adapter: harmful", {k: round(v, 3) for k, v in g_rem_h.items()},
          " harmless", {k: round(v, 3) for k, v in g_rem_b.items()})
    print("gate v·x  induce adapter: harmful", {k: round(v, 3) for k, v in g_ind_h.items()},
          " harmless", {k: round(v, 3) for k, v in g_ind_b.items()})

    # Axis: effective writes, signed by the gate on the prompts each adapter changes.
    w_rem = U_rem[:, 0] * g_rem_h["at_r_position"]
    w_ind = U_ind[:, 0] * g_ind_b["at_r_position"]
    cos = lambda a, b: float(t.nn.functional.cosine_similarity(a, b, dim=0))
    out["axis"] = {"cos_w_remove_w_induce": cos(w_rem, w_ind), "cos_w_remove_rhat": cos(w_rem, r_hat),
                   "cos_w_induce_rhat": cos(w_ind, r_hat)}
    print("axis:", {k: round(v, 3) for k, v in out["axis"].items()})

    # The inhibitor direction: the remove write with r̂ removed.
    inh = w_rem - (w_rem @ r_hat) * r_hat
    inh = inh / inh.norm()
    U_perp = (U_rem[:, 0] - (U_rem[:, 0] @ r_hat) * r_hat)[:, None]
    directions = {"r_hat": r_hat, "inhibitor": inh}

    # Mechanism: r̂'s projection through the layers, harmful prompts.
    def ablate_r():
        @contextlib.contextmanager
        def ctx():
            fw.intervention_applier.apply_direction_intervention(r, "ablate", layers=None)
            try:
                yield
            finally:
                fw.intervention_applier.clear_interventions()
        return ctx
    conditions = {"plain": nothing,
                  "adapter_full_u": lambda: with_adapter(layer, U_rem, V_rem),
                  "adapter_u_perp": lambda: with_adapter(layer, U_perp, V_rem),
                  "ablate_r_hat": ablate_r()}
    out["mechanism"] = {}
    for name, ctx in conditions.items():
        per_layer, _ = forward_stats(harmful, ctx, directions)
        with ctx():
            lo = ev._log_odds_metric(harmful)
        out["mechanism"][name] = {"proj_r_hat_by_layer": per_layer["r_hat"],
                                  "proj_inhibitor_by_layer": per_layer["inhibitor"], "refusal_log_odds": lo}
        sel = sorted({r.layer, min(r.layer + 2, n_layers - 1), min(r.layer + 4, n_layers - 1), min(r.layer + 8, n_layers - 1), n_layers - 1})
        print(f"  {name:16s} log-odds {lo:+6.2f} | r̂ proj at layers {sel}: "
              + " ".join(f"{per_layer['r_hat'][L]:+.2f}" for L in sel)
              + " | inhibitor proj: " + " ".join(f"{per_layer['inhibitor'][L]:+.2f}" for L in sel), flush=True)

    # Feature: the inhibitor direction in the plain model, harmful vs harmless.
    _, at_h = forward_stats(harmful, nothing, directions)
    _, at_b = forward_stats(harmless, nothing, directions)
    out["feature"] = {k: {"harmful_mean": float(t.tensor(at_h[k]).mean()), "harmless_mean": float(t.tensor(at_b[k]).mean()),
                          "auroc_harmful_gt_harmless": auroc(at_h[k], at_b[k])} for k in directions}
    print("feature (plain model, r̂'s layer/position):",
          {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out["feature"].items()})

    # Generality: does the inhibitor also suppress empathy?
    emp = get_concept("empathy")
    emp_pos, _ = emp.eval_data_fn()
    emp_ev = BigEvaluator(fw, detection_phrases=emp.detection_phrases, detection_fn=emp.detection_fn,
                          gen_batch_size=fw.evaluator.gen_batch_size)
    def detect_rate(evaluator, prompts, ctx):
        with ctx():
            texts = evaluator.generate_responses(prompts, max_new_tokens=64)
        return sum(map(evaluator._check_for_detection, texts)) / len(texts)
    out["generality"] = {
        "empathy_rate_plain": detect_rate(emp_ev, emp_pos, nothing),
        "empathy_rate_inhibitor": detect_rate(emp_ev, emp_pos, lambda: with_adapter(layer, U_perp, V_rem)),
        "refusal_rate_plain": detect_rate(ev, harmful, nothing),
        "refusal_rate_inhibitor": detect_rate(ev, harmful, lambda: with_adapter(layer, U_perp, V_rem)),
        "n_empathy": len(emp_pos)}
    print("generality:", {k: round(v, 3) for k, v in out["generality"].items()})

    path = f"results/finetune/{short}-inhibitor.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
