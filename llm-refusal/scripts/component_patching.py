"""Which attention / MLP outputs after a push carry its effect on refusal? (no generation)

The limitation push works at the chat template's tokens (RESULTS §39), within ~3 layers of where it is added (§37),
while r̂ is still read downstream (§34). For each condition (`sub:<stem>` / `add:<stem>`, at the stem's layer,
strength via `STEM*s`, every prompt token), two runs are cached on each batch: clean and pushed. Then for each of the
--window layers from the push layer on, and each sublayer (attn, mlp), one output is swapped between them:

  necessity    the pushed run with that sublayer's output taken from the clean run: how much of the push's
               log-odds effect is lost (1 = all of it, 0 = none)
  sufficiency  the clean run with that sublayer's output taken from the pushed run: how much of the effect it
               brings by itself

each at every token (`all`) and at the template suffix only (`template`, push_positions.py's mask). Read by the
refusal log-odds at the generation boundary, on harmful prompts for sub: and harmless for add:.

  .venv/bin/python3 llm-refusal/scripts/component_patching.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --conditions lim=sub:results/limits4-Llama-3.1-8B-Instruct-direction

Writes results/analysis/<short>-component-patching[-tag].json and prints the per-layer shares.
"""
import argparse
import math
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

import prompts  # noqa: E402
from formatting import last_real_token_indices  # noqa: E402
from interventions import get_sublayers  # noqa: E402
from jbb_safety import split_strength  # noqa: E402
from probe import save_json  # noqa: E402
from push_positions import position_masks  # noqa: E402

COMPONENTS = ["attn", "mlp"]
MODES = ["necessity", "sufficiency"]
POSITIONS = ["all", "template"]


def effect_share(mode: str, patched: float, clean: float, pushed: float) -> float:
    """necessity: the fraction of the push's effect lost by patching the clean output into the pushed run;
    sufficiency: the fraction recovered by patching the pushed output into the clean run. NaN when the push moves
    the log-odds by < 0.5 nats."""
    gap = pushed - clean
    if abs(gap) < 0.5:
        return float("nan")
    return (pushed - patched) / gap if mode == "necessity" else (patched - clean) / gap


def _out(output):
    return output[0] if isinstance(output, tuple) else output


def _replace(output, new):
    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


@t.no_grad()
def run_batch(run, enc, push: Optional[t.Tensor], layer: int, capture: Dict = None,
              patch: Optional[tuple] = None) -> List[float]:
    """One forward pass: `push` added at `layer`'s input on every token; `capture` (dict (l, comp) → None) is filled
    with those sublayer outputs; `patch` = ((l, comp), source [B, T, d], mask [B, T]) replaces that output where mask.
    Returns per-prompt refusal log-odds."""
    dev = run.model.device
    handles = []
    try:
        if push is not None:
            v = push.to(dev, run.model.dtype)
            handles.append(run.blocks[layer].register_forward_pre_hook(
                lambda m, args: (args[0] + v,) + args[1:]))
        for (l, comp) in (capture or {}):
            mod = get_sublayers(run.blocks[l])[COMPONENTS.index(comp)]

            def grab(m, i, o, key=(l, comp)):
                capture[key] = _out(o).detach().clone()
            handles.append(mod.register_forward_hook(grab))
        if patch is not None:
            (l, comp), src, mask = patch
            mod = get_sublayers(run.blocks[l])[COMPONENTS.index(comp)]
            mk = mask.to(dev)[..., None]

            def swap(m, i, o):
                h = _out(o)
                return _replace(o, t.where(mk, src.to(h.dtype), h))
            handles.append(mod.register_forward_hook(swap))
        logits = run.model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                           position_ids=enc["position_ids"].to(dev)).logits
    finally:
        for h in handles:
            h.remove()
    last = last_real_token_indices(enc["attention_mask"])
    return [run.ev.metric.compute_log_odds(logits[j, last[j]].float()) for j in range(len(last))]


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", required=True, help="comma-separated name=sub:<stem>[*s] or name=add:<stem>[*s]")
    ap.add_argument("--window", type=int, default=8, help="layers from the push layer on")
    ap.add_argument("--n", type=int, default=64, help="prompts")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (a smoke check while the GPU is busy)")
    a = ap.parse_args()

    from datatypes import DirectionVector
    from probe import load_run
    run = load_run(a.model, direction=False, **({"force_cpu": True} if a.cpu else {}))
    harmful, harmless = (ps[:a.n] for ps in prompts.create_refusal_eval_data())
    n_suffix = run.fmt.assistant_prefix_tokens
    out = {"model": a.model, "n_suffix": n_suffix, "window": a.window, "conditions": {}}
    for name, spec in (c.split("=", 1) for c in a.conditions.split(",")):
        kind, _, stem = spec.partition(":")
        assert kind in ("sub", "add"), f"{name}: only sub:/add: conditions"
        stem, strength = split_strength(stem)
        d = DirectionVector.load(stem)
        push = d.vector.float().cpu() * (-strength if kind == "sub" else strength)
        layers = list(range(d.layer, min(d.layer + a.window, len(run.blocks))))
        keys = [(l, c) for l in layers for c in COMPONENTS]
        ps = harmful if kind == "sub" else harmless
        clean_s, pushed_s = [], []
        patched = {(m, p, l, c): [] for m in MODES for p in POSITIONS for (l, c) in keys}
        for i in range(0, len(ps), a.batch_size):
            enc = run.fmt.format_batch(ps[i:i + a.batch_size])
            masks = position_masks(enc["attention_mask"], n_suffix)
            cache_c, cache_p = dict.fromkeys(keys), dict.fromkeys(keys)
            clean_s += run_batch(run, enc, None, d.layer, capture=cache_c)
            pushed_s += run_batch(run, enc, push, d.layer, capture=cache_p)
            for p in POSITIONS:
                for key in keys:
                    patched[("necessity", p) + key] += run_batch(run, enc, push, d.layer,
                                                                 patch=(key, cache_c[key], masks[p]))
                    patched[("sufficiency", p) + key] += run_batch(run, enc, None, d.layer,
                                                                   patch=(key, cache_p[key], masks[p]))
        clean, pushed = mean(clean_s), mean(pushed_s)
        c = {"spec": spec, "layer": d.layer, "clean": clean, "pushed": pushed, "layers": layers, "share": {}}
        for (m, p, l, comp), s in patched.items():
            c["share"].setdefault(m, {}).setdefault(p, {}).setdefault(comp, []).append(
                effect_share(m, mean(s), clean, pushed))
        out["conditions"][name] = c
        print(f"\n{name} ({spec}) L{d.layer}: clean {clean:+.2f} → pushed {pushed:+.2f}", flush=True)
        for m in MODES:
            for p in POSITIONS:
                for comp in COMPONENTS:
                    print(f"  {m:11s} {p:8s} {comp:4s} " + " ".join(
                        f"L{l}:{x:+.2f}" for l, x in zip(layers, c["share"][m][p][comp])), flush=True)
    print("saved", save_json(f"results/analysis/{run.short}-component-patching{'-' + a.tag if a.tag else ''}.json",
                             out))


if __name__ == "__main__":
    main()
