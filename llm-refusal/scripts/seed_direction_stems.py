"""Save per-seed difference-in-means vectors from regrown_seed_directions.py as direction stems (no model load).

For ablate_in_variants.py: is a cross-seed transfer gap a layer effect or a direction effect? Writes, for each
--variants entry and --layers L, the raw regrown difference-in-means ("raw") and, for regrown seeds, what refusal
training added over the benign twin ("delta", readers-r32[-sS] minus readers-r0[-sS]) as
results/seeddir-<short>-<variant>-<raw|delta>-L<L>-P<pos>-direction.{pt,json}, scaled to unit norm (ablation
ignores the norm). Prints each stem's cosine with --compare stems at the same layer index, which also checks the
layer convention against the search's own directions.

  .venv/bin/python3 llm-refusal/scripts/seed_direction_stems.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --variants readers-r32,readers-r32-s1,readers-r32-s2 --layers 29,30 \\
      --compare results/regrown100-Meta-Llama-3-8B-Instruct-refusal-direction
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported

import torch as t  # noqa: E402

from datatypes import DirectionVector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--pos", type=int, default=-1)
    ap.add_argument("--compare", default="", help="comma-separated direction stems to report cosines against")
    a = ap.parse_args()
    short = a.model.split("/")[-1]
    vecs = t.load(f"results/analysis/{short}-regrown-seed-directions.pt")
    refs = {s: DirectionVector.load(s) for s in a.compare.split(",") if s}
    cos = lambda x, y: float(x.float() @ y.float() / (x.float().norm() * y.float().norm()))  # noqa: E731
    for v in a.variants.split(","):
        kinds = {"raw": vecs[v]}
        twin = "-".join([v.split("-")[0], "r0", *v.split("-")[2:]])
        if twin != v and twin in vecs:
            kinds["delta"] = vecs[v] - vecs[twin]
        for kind, m in kinds.items():
            for L in (int(x) for x in a.layers.split(",")):
                d = m[L] / m[L].norm()
                stem = f"results/seeddir-{short}-{v}-{kind}-L{L}-P{a.pos}-direction"
                DirectionVector(vector=d.to(t.bfloat16), layer=L, position_index=a.pos, score=0).save(stem)
                rel = " ".join(f"cos {os.path.basename(s)[:24]}(L{r.layer})={cos(d, r.vector):+.2f}" for s, r in refs.items())
                print(f"{stem}  {rel}", flush=True)


if __name__ == "__main__":
    main()
