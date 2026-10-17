"""Save a plain difference-of-means direction (or several concepts' mean of them) at a fixed layer and position.

The limitation mean (`limits4`) is the average of four search-selected directions. Is the jailbreak a property of
that selection and those prompts, or of the concept? For each --concepts member, the mean residual of its training
positives minus its negatives at --like's layer and position (`probe.residuals_at`), unit-normed; the members are
averaged and the result is rescaled to --like's norm and saved at --like's layer and position, for every --halves
entry: `all` (every training pair), `a` (the first half of each concept's pairs), `b` (the second half). `{half}` in
--out is replaced by the half's name.

  .venv/bin/python3 llm-refusal/scripts/save_diffmean.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --concepts inability realtime memory persona --like results/limits4-Llama-3.1-8B-Instruct-direction \\
      --out "results/dm{half}-Llama-3.1-8B-Instruct-direction"

Prints each saved direction's cosine with --like and with each other.
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402


def half(pairs: Tuple[List[str], List[str]], which: str) -> Tuple[List[str], List[str]]:
    """`all`, or the first (`a`) / second (`b`) half of a concept's positives and negatives (pairs are aligned)."""
    pos, neg = pairs
    k = len(pos) // 2
    return {"all": (pos, neg), "a": (pos[:k], neg[:k]), "b": (pos[k:], neg[k:])}[which]


def mean_of_units(vs: List[t.Tensor], norm: float) -> t.Tensor:
    u = t.stack([v / v.norm() for v in vs]).mean(0)
    return u / u.norm() * norm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concepts", nargs="+", required=True)
    ap.add_argument("--like", required=True, help="stem whose layer, position and norm the result takes")
    ap.add_argument("--halves", default="all,a,b")
    ap.add_argument("--out", required=True, help="output stem; {half} is replaced by the half's name")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (a smoke check while the GPU is busy)")
    a = ap.parse_args()

    from concept import get_concept
    from datatypes import DirectionVector
    from probe import load_run, residuals_at
    like = DirectionVector.load(a.like)
    norm = float(like.vector.float().norm())
    run = load_run(a.model, direction=False, **({"force_cpu": True} if a.cpu else {}))
    acts = lambda ps: residuals_at(run.model, run.fmt, run.blocks, ps, like.position_index)[:, like.layer]
    saved: Dict[str, t.Tensor] = {}
    for h in a.halves.split(","):
        diffs = []
        for c in a.concepts:
            pos, neg = half(get_concept(c).train_data_fn(), h)
            diffs.append(acts(pos).mean(0) - acts(neg).mean(0))
        v = mean_of_units(diffs, norm)
        out = a.out.replace("{half}", h)
        DirectionVector(vector=v, layer=like.layer, position_index=like.position_index, score=0.0).save(out)
        saved[h] = v / v.norm()
        print(f"saved {out} (layer {like.layer}, norm {norm:.2f}); cos with --like "
              f"{float(saved[h] @ (like.vector.float().cpu() / like.vector.float().norm())):+.3f}", flush=True)
    names = list(saved)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            print(f"cos {x} / {y} {float(saved[x] @ saved[y]):+.3f}")


if __name__ == "__main__":
    main()
