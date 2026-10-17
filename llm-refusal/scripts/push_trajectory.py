"""Does a push change how strongly the refusal direction is read downstream? (no generation)

Subtracting the limitation mean removes refusal without r̂ in it (RESULTS §32). Either the push lowers the r̂ reading
that the later layers act on (the limitation feature feeds refusal), or refusal is still read and its output is
blocked further on. For each condition (jbb_safety.py's grammar: `baseline`, `sub:<stem>`, `add:<stem>`,
`ablate:<spec>`, `edit:<spec>`), on the refusal eval set's harmful and harmless prompts: the mean projection onto r̂
at r̂'s position at every layer, and the refusal log-odds. Per layer the harmful mean is placed on the baseline's
harmless → harmful scale (0 = the baseline harmless mean, 1 = the baseline harmful mean), as is the harmless mean.
`direct` is the share a sub:/add: vector moves r̂'s reading by itself (its own r̂ component on the same scale, at its
layer and after): anything beyond it is mediated by the layers in between.

  .venv/bin/python3 llm-refusal/scripts/push_trajectory.py --model meta-llama/Llama-3.1-8B-Instruct --tag lim \\
      --conditions baseline=baseline,lim=sub:results/limits4-Llama-3.1-8B-Instruct-direction,limadd=add:results/limits4-Llama-3.1-8B-Instruct-direction

Writes results/analysis/<short>-push-trajectory-<tag>.json and prints the scale at r̂'s layer and the last layer.
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

import prompts  # noqa: E402
from datatypes import DirectionVector  # noqa: E402
from jbb_safety import condition_context, split_strength  # noqa: E402
from probe import load_run, residuals_at, save_json  # noqa: E402


def on_scale(x: List[float], lo: List[float], hi: List[float]) -> List[float]:
    """Per layer, x on the lo → hi scale (0 = lo, 1 = hi); NaN where the gap is ~0."""
    return [(a - b) / (c - b) if abs(c - b) > 1e-6 else float("nan") for a, b, c in zip(x, lo, hi)]


def direct_share(spec: str, r_hat: t.Tensor, lo: List[float], hi: List[float]) -> Optional[List[float]]:
    """For sub:/add: of a saved stem, the vector's own signed r̂ component per layer on the lo → hi gap scale
    (0 before its layer, where it has not been added yet); None for other kinds."""
    kind, _, stem = spec.partition(":")
    if kind not in ("sub", "add"):
        return None
    stem, strength = split_strength(stem)
    d = DirectionVector.load(stem)
    c = float(d.vector.float().cpu() @ r_hat) * (-strength if kind == "sub" else strength)
    return [c / (h - l) if l_i >= d.layer and abs(h - l) > 1e-6 else 0.0
            for l_i, (l, h) in enumerate(zip(lo, hi))]


def fmt_lo(x: Optional[float]) -> str:
    return "   n/a" if x is None else f"{x:+6.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", required=True, help="comma-separated name=kind[:spec]; must include baseline")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100, help="prompts per side")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (a smoke check while the GPU is busy)")
    a = ap.parse_args()
    conds = dict(c.split("=", 1) for c in a.conditions.split(","))
    assert conds.get("baseline") == "baseline", "the conditions need baseline=baseline (the scale's end points)"

    run = load_run(a.model, **({"force_cpu": True} if a.cpu else {}))
    harmful, harmless = (ps[:a.n] for ps in prompts.create_refusal_eval_data())
    r_hat, pos = run.r_hat, run.r.position_index
    proj = lambda ps: (residuals_at(run.model, run.fmt, run.blocks, ps, pos) @ r_hat).mean(0).tolist()   # [L]

    raw: Dict[str, dict] = {}
    for name, spec in conds.items():
        with condition_context(run, spec):
            raw[name] = {"spec": spec, "harmful": proj(harmful), "harmless": proj(harmless),
                         "log_odds_harmful": run.ev._log_odds_metric(harmful),
                         "log_odds_harmless": run.ev._log_odds_metric(harmless)}
        print(f"{name}: done", flush=True)

    lo, hi = raw["baseline"]["harmless"], raw["baseline"]["harmful"]
    L, last = run.r.layer, len(lo) - 1
    out = {"model": a.model, "r_layer": L, "r_pos": pos, "n_harmful": len(harmful), "n_harmless": len(harmless),
           "conditions": {}}
    print(f"\nr̂ at L{L}; harmful / harmless on the baseline harmless → harmful scale, at L{L} and L{last}; log-odds")
    for name, c in raw.items():
        s_h, s_n = on_scale(c["harmful"], lo, hi), on_scale(c["harmless"], lo, hi)
        direct = direct_share(c["spec"], r_hat, lo, hi)
        out["conditions"][name] = {**c, "scale_harmful": s_h, "scale_harmless": s_n, "direct": direct}
        dtxt = f"  direct {direct[L]:+.2f} / {direct[last]:+.2f}" if direct else ""
        print(f"{name:12s} harmful {s_h[L]:5.2f} / {s_h[last]:5.2f}   harmless {s_n[L]:5.2f} / {s_n[last]:5.2f}   "
              f"log-odds {fmt_lo(c['log_odds_harmful'])} / {fmt_lo(c['log_odds_harmless'])}{dtxt}")
    print("saved", save_json(f"results/analysis/{run.short}-push-trajectory-{a.tag}.json", out))


if __name__ == "__main__":
    main()
