"""Which tokens does a push act through? (no generation)

Subtracting the limitation mean at its layer removes refusal, and the same push three layers later does nothing
(RESULTS §37), while r̂ is still read downstream (§34). If the push changes how the request itself is represented, it
should work when applied only to the instruction's tokens (later layers then read the change through attention);
if it acts where refusal is computed, only at the template tokens after the instruction (Arditi's post-instruction
positions). Each condition (`sub:<stem>` / `add:<stem>`, at the stem's layer, strength 1) is applied to one token
subset of the prompt at a time:

  all          every real token (the usual push, prefill only)
  instruction  every real token before the chat template's assistant suffix (BOS, user header, the request)
  template     the assistant suffix (formatting.assistant_prefix_tokens tokens, ending at the generation boundary)
  last         the final token only

and the refusal log-odds at the generation boundary is read on the refusal eval set's harmful and harmless prompts.
`share` = (subset − baseline) / (all − baseline): how much of the full push's effect that subset carries.

  .venv/bin/python3 llm-refusal/scripts/push_positions.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --conditions lim=sub:results/limits4-Llama-3.1-8B-Instruct-direction,limadd=add:results/limits4-Llama-3.1-8B-Instruct-direction

Writes results/analysis/<short>-push-positions[-tag].json.
"""
import argparse
import math
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

import prompts  # noqa: E402
from formatting import last_real_token_indices  # noqa: E402
from probe import save_json  # noqa: E402

SUBSETS = ["all", "instruction", "template", "last"]


def position_masks(attention_mask: t.Tensor, n_suffix: int) -> Dict[str, t.Tensor]:
    """Bool [B, T] masks of each subset's token positions in a right-padded batch."""
    last = last_real_token_indices(attention_mask)[:, None]          # [B, 1]
    pos = t.arange(attention_mask.shape[1])[None]                    # [1, T]
    real = attention_mask.bool()
    template = (pos > last - n_suffix) & (pos <= last)
    return {"all": real, "instruction": real & ~template, "template": template, "last": pos == last}


def share(subset: float, full: float, base: float) -> float:
    """The subset's effect as a fraction of the full push's (NaN when the full push moves < 0.5 nats)."""
    return (subset - base) / (full - base) if abs(full - base) >= 0.5 else float("nan")


@t.no_grad()
def log_odds(run, ps: List[str], vec: Optional[t.Tensor], layer: Optional[int], subset: str,
             batch_size: int = 16) -> float:
    """Mean refusal log-odds at the generation boundary, with `vec` added at `layer`'s input on `subset`'s tokens."""
    n_suffix = run.fmt.assistant_prefix_tokens
    scores: List[float] = []
    for i in range(0, len(ps), batch_size):
        enc = run.fmt.format_batch(ps[i:i + batch_size])
        dev = run.model.device
        handle = None
        if vec is not None:
            mask = position_masks(enc["attention_mask"], n_suffix)[subset].to(dev)
            v = vec.to(dev, run.model.dtype)

            def hook(module, args):
                h = args[0]
                return (h + mask[..., None].to(h.dtype) * v,) + args[1:]
            handle = run.blocks[layer].register_forward_pre_hook(hook)
        try:
            logits = run.model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                               position_ids=enc["position_ids"].to(dev)).logits
        finally:
            if handle is not None:
                handle.remove()
        last = last_real_token_indices(enc["attention_mask"])
        scores += [run.ev.metric.compute_log_odds(logits[j, last[j]].float()) for j in range(len(last))]
    finite = [s for s in scores if not math.isnan(s)]
    return sum(finite) / len(finite)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", required=True, help="comma-separated name=sub:<stem> or name=add:<stem>")
    ap.add_argument("--n", type=int, default=100, help="prompts per side")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (a smoke check while the GPU is busy)")
    a = ap.parse_args()

    from datatypes import DirectionVector
    from probe import load_run
    run = load_run(a.model, direction=False, **({"force_cpu": True} if a.cpu else {}))
    harmful, harmless = (ps[:a.n] for ps in prompts.create_refusal_eval_data())
    sets = {"harmful": harmful, "harmless": harmless}
    base = {k: log_odds(run, ps, None, None, "all") for k, ps in sets.items()}
    out = {"model": a.model, "n_suffix": run.fmt.assistant_prefix_tokens, "baseline": base, "conditions": {}}
    print(f"suffix tokens {out['n_suffix']}; baseline log-odds harmful {base['harmful']:+.2f} "
          f"harmless {base['harmless']:+.2f}", flush=True)
    for name, spec in (c.split("=", 1) for c in a.conditions.split(",")):
        kind, _, stem = spec.partition(":")
        assert kind in ("sub", "add"), f"{name}: only sub:/add: conditions"
        d = DirectionVector.load(stem)
        vec = d.vector.float().cpu() * (-1 if kind == "sub" else 1)
        c = {"spec": spec, "layer": d.layer, "log_odds": {}, "share": {}}
        for k, ps in sets.items():
            c["log_odds"][k] = {s: log_odds(run, ps, vec, d.layer, s) for s in SUBSETS}
            c["share"][k] = {s: share(c["log_odds"][k][s], c["log_odds"][k]["all"], base[k]) for s in SUBSETS}
        out["conditions"][name] = c
        key = "harmful" if kind == "sub" else "harmless"
        print(f"{name:10s} L{d.layer:<3d} {key:8s} " + "  ".join(
            f"{s} {c['log_odds'][key][s]:+6.2f} ({c['share'][key][s]:.2f})" for s in SUBSETS), flush=True)
    print("saved", save_json(f"results/analysis/{run.short}-push-positions{'-' + a.tag if a.tag else ''}.json", out))


if __name__ == "__main__":
    main()
