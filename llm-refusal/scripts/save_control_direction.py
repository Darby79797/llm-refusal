"""Save a control version of a saved direction as an ordinary stem (no model), for `--direction-file`:

  --random-like SEED     a random direction with the stem's norm, layer and position (is any push this size enough?)
  --project-out STEM...  the stem with those directions' span projected out, rescaled to its norm (does the push work
                         without the part it shares with them?)
  --like STEM            the stem's direction at STEM's layer and norm (another learned direction, pushed as hard and
                         where STEM is)
  --layer N              the stem at layer N, its norm kept (the same push at another layer)

  .venv/bin/python3 llm-refusal/scripts/save_control_direction.py results/limits4-Qwen2.5-7B-Instruct-direction \\
      --project-out results/Qwen2.5-7B-Instruct-unknown-direction --out results/limits4perpunk-Qwen2.5-7B-Instruct-direction

Prints the result's cosine with the stem and with each projected-out direction.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported

import dirspecs  # noqa: E402
from datatypes import DirectionVector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--random-like", type=int, metavar="SEED")
    g.add_argument("--project-out", nargs="+", metavar="STEM")
    g.add_argument("--like", metavar="STEM")
    g.add_argument("--layer", type=int)
    ap.add_argument("--out", required=True, help="output stem (no .pt/.json)")
    a = ap.parse_args()
    if a.random_like is not None:
        d = dirspecs.save_random_like(a.stem, a.out, a.random_like)
    elif a.project_out:
        d = dirspecs.save_projected_out(a.stem, a.project_out, a.out)
    else:
        d = dirspecs.save_moved(a.stem, a.out, layer=a.layer, like=a.like)
    u = dirspecs.unit(d.vector.float())
    for x in [a.stem] + (a.project_out or []) + ([a.like] if a.like else []):
        print(f"cos {float(dirspecs.unit(DirectionVector.load(x).vector.float()) @ u):+.3f}  {x}")
    print(f"saved {a.out} (layer {d.layer}, norm {float(d.vector.float().norm()):.2f})")


if __name__ == "__main__":
    main()
