"""Save a copy of a saved direction with some coordinates zeroed (renormalised to the same norm).

Qwen's r̂ loads 1-3% of its squared norm on the low-RMSNorm-gain outlier dimensions
(results/analysis/referee-geometry.json); zeroing them tests whether the edit cost and the
late shared rise come from those coordinates.

  .venv/bin/python3 llm-refusal/scripts/mask_direction.py results/Qwen2.5-0.5B-Instruct-refusal-direction --dims 490 --out results/Qwen2.5-0.5B-Instruct-refusal-nomassive-direction
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch as t  # noqa: E402

from datatypes import DirectionVector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem")
    ap.add_argument("--dims", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = DirectionVector.load(a.stem)
    v = d.vector.clone().float()
    before = float(v.norm())
    share = float((v[a.dims] ** 2).sum() / (v ** 2).sum())
    v[a.dims] = 0
    v = v * before / v.norm()
    cos = float(t.nn.functional.cosine_similarity(v, d.vector.float(), dim=0))
    DirectionVector(vector=v.to(d.vector.dtype), layer=d.layer, position_index=d.position_index, score=d.score).save(a.out)
    print(f"zeroed dims {a.dims}: {share:.3%} of ||r||^2; cos with the original {cos:.4f}; saved {a.out}")


if __name__ == "__main__":
    main()
