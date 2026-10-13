"""Save principal axes of a direction pool as ordinary direction stems (no model; CPU, seconds).

`pc:K@POOL` specs work in `ablate_in_variants.py`, `late_readout.py` and `jbb_safety.py`, but scripts that take a
plain stem (`edit_cost_sweep.py --direction`, `--extra-edit-directions`) need a file. This writes each requested axis
as results/pcax-<short>-PC<k>-direction: unit norm, signed so the pool's harmful-minus-harmless contrasts project
positively on average, at the layer of the pool's last stem (or --layer) and position -1.

  .venv/bin/python3 llm-refusal/scripts/save_pc_axes.py --model Qwen/Qwen2.5-3B-Instruct --ks 1 2 \\
      --pool "perp:results/cleanall-Qwen2.5-3B-Instruct-L22-P-1-direction+perp:..."

The JSON beside each .pt records only layer/position/score (DirectionVector's format); the pool is printed and
written to <out-prefix>-axes.json.
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True, help="'+'-joined specs, as after the @ of a pc: spec")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--layer", type=int, default=None, help="recorded layer (default: the pool's last stem's)")
    ap.add_argument("--out-prefix", default=None, help="default results/pcax-<short>")
    a = ap.parse_args()
    short = a.model.split("/")[-1]
    r = DirectionVector.load(f"results/{short}-refusal-direction")
    last = dirspecs.load_spec(a.pool.split("+")[-1], r)
    layer = a.layer if a.layer is not None else getattr(last, "layer", None)
    if layer is None:
        sys.exit("the pool's last spec has no layer: pass --layer")
    for s in dirspecs.save_pc_axes(a.pool, r, a.ks, layer, a.out_prefix or f"results/pcax-{short}"):
        print("saved", s)


if __name__ == "__main__":
    main()
