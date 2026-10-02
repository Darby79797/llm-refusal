"""Is the orthogonalised model the hook-ablated model? A teacher-forced check.

Generating 512 tokens per condition and classifying them with LlamaGuard compares
one greedy sample per prompt, and greedy decoding turns bf16 rounding into
different texts (on Qwen2.5-0.5B the hook and weight versions share a median of
48 tokens before diverging). This compares whole next-token distributions
instead, on text that was already generated and safety-scored: an earlier safety
run's saved 512-token responses.

For each saved response (the hook-ablated ones the safety score was measured on,
and the unablated baseline ones), one forward pass under each of:

  hook      directional ablation with hooks (the reference)
  ortho     the direction orthogonalised out of the weights (orthogonalize.py)
  hook_bs1  hook ablation again, one row at a time: bf16 batch-shape noise, the
            floor that `ortho` can't be expected to beat
  clean     no intervention: how far ablation moves the model at all, for scale
  hook_fp32 (--fp32-reference) hook ablation on a float32 copy of the model: the
            precision noise the project already accepts (bf16 vs fp32 changed
            7/1361 refusal labels). Needs a second, full-precision copy in memory,
            so only for small models. The batch-shape floor alone is too tight a
            reference: rows without padding are batch-invariant in bf16 (KL = 0),
            while the weight edit rounds every edited weight to bf16 once.

and at every response token: KL(hook || x), top-1 agreement with hook, and the
NLL of the saved token. If ortho vs hook sits at the hook_bs1 noise floor, far
below clean vs hook, the edited model has the hook model's output distribution,
so the hook runs' safety scores carry over to it. Also reports the refusal score
(log-odds, no decoding) on the prompts under hook and ortho.

    .venv/bin/python3 llm-refusal/scripts/ortho_equivalence.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --generations results/sweep5-Meta-Llama-3-8B-Instruct-refusal_arditi_exact-evaluate-L12-P-5-T512-generations.json

Writes results/ortho/<model>-equivalence.json.
"""
import argparse
import contextlib
import json
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
_high = os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(min(0.6, 0.75 * float(_high))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hf_offline import use_offline_if_cached  # noqa: E402
# Before transformers is imported: cached models then load without Hub calls.
use_offline_if_cached(next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--model"), None))

import torch as t  # noqa: E402

from datatypes import DirectionVector  # noqa: E402
from framework import DirectionTestFramework  # noqa: E402
from orthogonalize import orthogonalized, prepare_edit  # noqa: E402

VARIANTS = ["ortho", "hook_bs1", "clean"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--generations", required=True, help="a T512 evaluate generations JSON with global_ablation")
    ap.add_argument("--concept", default="refusal_arditi_exact")
    ap.add_argument("--direction", default=None,
                    help="direction path without .pt/.json (default: results/<model>-<concept>-direction)")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="rows per forward; each variant's fp32 log-probs are rows x tokens x vocab")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--fp32-reference", choices=["auto", "on", "off"], default="auto",
                    help="also compare with hook ablation on a float32 copy (auto: models <= 4B params, "
                         "whose fp32 copy fits beside the bf16 one)")
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    fw = DirectionTestFramework(model_name=args.model, concept=args.concept)
    model, fmt = fw.model, fw.prompt_formatter
    direction = DirectionVector.load(args.direction or f"results/{short}-{args.concept}-direction")
    with open(args.generations) as f:
        gens = json.load(f)
    applier = fw.intervention_applier
    model32 = applier32 = None
    n_params = sum(p.numel() for p in model.parameters())
    use_fp32 = args.fp32_reference == "on" or (args.fp32_reference == "auto" and n_params <= 4e9)
    if use_fp32:
        from transformers import AutoModelForCausalLM
        from interventions import ModelInterventionApplier
        model32 = AutoModelForCausalLM.from_pretrained(args.model, dtype=t.float32).to(model.device).eval()
        applier32 = ModelInterventionApplier(model32)

    def hooks_on(app):
        @contextlib.contextmanager
        def ctx():
            app.apply_direction_intervention(direction, "ablate", layers=None)
            try:
                yield
            finally:
                app.clear_interventions()
        return ctx
    hooked = hooks_on(applier)

    def log_probs(enc, ctx, rows=None, m=None):
        """fp32 log-probs at every scored position of `enc` (optionally only `rows`)."""
        m = m or model
        sl = slice(None) if rows is None else rows
        dev = model.device
        with ctx(), t.no_grad():
            logits = m(input_ids=enc["input_ids"][sl].to(dev), attention_mask=enc["attention_mask"][sl].to(dev),
                           position_ids=enc["position_ids"][sl].to(dev), use_cache=False).logits
        scored = enc["labels"][sl][:, 1:] != -100
        return t.log_softmax(logits[:, :-1][scored.to(dev)].float(), dim=-1)

    if use_fp32:
        VARIANTS.append("hook_fp32")
    out = {"model": args.model, "concept": args.concept, "generations": args.generations,
           "direction": {"layer": direction.layer, "position_index": direction.position_index},
           "batch_size": args.batch_size, "dtype": str(model.dtype), "max_tokens": args.max_tokens, "sets": {}}
    # Built once and swapped in per batch (rebuilding it is ~2 s a batch on an 8B model).
    edit = prepare_edit(model, direction.vector)
    start = time.time()
    for set_name in ("global_ablation", "baseline"):
        entries = gens[set_name]
        prompts, responses = [e["prompt"] for e in entries], [e["response"] for e in entries]
        sums = {v: {"kl": 0.0, "top1": 0, "nll": 0.0} for v in (*VARIANTS, "hook")}
        kl_max = {v: 0.0 for v in VARIANTS}
        n_tok = 0
        for i in range(0, len(entries), args.batch_size):
            enc = fmt.format_with_completions(prompts[i:i + args.batch_size], responses[i:i + args.batch_size],
                                              max_completion_tokens=args.max_tokens)
            targets = enc["labels"][:, 1:][enc["labels"][:, 1:] != -100].to(model.device)
            ref = log_probs(enc, hooked)
            ref_top = ref.argmax(-1)
            sums["hook"]["nll"] -= ref.gather(-1, targets[:, None]).sum().item()
            # One variant's log-probs at a time beside the reference (each is
            # rows x tokens x vocab in fp32: ~1 GB per 2 rows of 512 tokens on Llama-3).
            others = {
                "ortho": lambda: log_probs(enc, lambda: orthogonalized(model, direction.vector, prepared=edit)),
                "hook_bs1": lambda: t.cat([log_probs(enc, hooked, rows=slice(j, j + 1))
                                           for j in range(enc["input_ids"].shape[0])]),
                "clean": lambda: log_probs(enc, contextlib.nullcontext),
            }
            if model32 is not None:
                others["hook_fp32"] = lambda: log_probs(enc, hooks_on(applier32), m=model32)
            for v, compute in others.items():
                lp = compute()
                kl = (ref.exp() * (ref - lp)).sum(-1)
                sums[v]["kl"] += kl.sum().item()
                kl_max[v] = max(kl_max[v], kl.max().item())
                sums[v]["top1"] += int((lp.argmax(-1) == ref_top).sum())
                sums[v]["nll"] -= lp.gather(-1, targets[:, None]).sum().item()
                del lp, kl
            n_tok += len(targets)
            print(f"  {set_name}: {min(i + args.batch_size, len(entries))}/{len(entries)} rows, "
                  f"{n_tok} tokens ({time.time() - start:.0f}s)", flush=True)
        res = {"n_rows": len(entries), "n_tokens": n_tok, "nll_hook": sums["hook"]["nll"] / n_tok}
        for v in VARIANTS:
            res[v] = {"kl_mean": sums[v]["kl"] / n_tok, "kl_max": kl_max[v],
                      "top1_agreement": sums[v]["top1"] / n_tok, "nll": sums[v]["nll"] / n_tok}
        out["sets"][set_name] = res
        print(json.dumps({set_name: res}, indent=1), flush=True)

    prompts = [e["prompt"] for e in gens["global_ablation"]]
    with hooked():
        lo_hook = fw.evaluator._log_odds_metric(prompts)
    with orthogonalized(model, direction.vector, prepared=edit):
        lo_ortho = fw.evaluator._log_odds_metric(prompts)
    out["refusal_log_odds"] = {"clean": fw.evaluator._log_odds_metric(prompts), "hook": lo_hook, "ortho": lo_ortho}
    print("refusal log-odds:", out["refusal_log_odds"])
    out["seconds"] = time.time() - start

    os.makedirs(os.path.join("results", "ortho"), exist_ok=True)
    path = os.path.join("results", "ortho", f"{short}-equivalence.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
