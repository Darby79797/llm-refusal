"""Save the mean of several saved directions (unit-normed, then rescaled to their mean norm) as one stem (no model).

For a "shared" direction across concepts, e.g. the assistant-limitation concepts:

  .venv/bin/python3 llm-refusal/scripts/save_mean_direction.py --out results/limits-Qwen2.5-7B-Instruct-direction \\
      results/Qwen2.5-7B-Instruct-inability-direction results/Qwen2.5-7B-Instruct-realtime-direction

Prints each member's cosine with the mean.
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
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--out", required=True, help="output stem (no .pt/.json)")
    ap.add_argument("--layer", type=int, default=None, help="recorded layer (default: the first stem's)")
    a = ap.parse_args()
    m = dirspecs.save_mean_direction(a.stems, a.out, a.layer)
    u = dirspecs.unit(m.vector.float())
    for x in a.stems:
        print(f"cos {float(dirspecs.unit(DirectionVector.load(x).vector.float()) @ u):+.3f}  {x}")
    print(f"saved {a.out} (layer {m.layer}, norm {float(m.vector.float().norm()):.2f})")


if __name__ == "__main__":
    main()
