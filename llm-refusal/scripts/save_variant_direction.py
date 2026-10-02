"""Save the difference-in-means direction of a model VARIANT (edited and/or adapted) at given
coordinates, so it can be evaluated inside that variant with --direction-file.

  .venv/bin/python3 llm-refusal/scripts/save_variant_direction.py --model Qwen/Qwen2.5-0.5B-Instruct \\
      --adapter results/finetune/Qwen2.5-0.5B-Instruct-refusal-regrow-readers-r32 --orthogonalize-first \\
      --coords 9:-1 11:-1 12:-1 16:-1 18:-1 --out-prefix results/regrownvar-Qwen2.5-0.5B-Instruct
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

from datatypes import DirectionVector, PromptData  # noqa: E402
from probe import load_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--adapter")
    ap.add_argument("--orthogonalize-first", action="store_true")
    ap.add_argument("--coords", nargs="+", required=True, help="layer:pos pairs")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    run = load_run(a.model, a.concept)
    cfg = {"orthogonalize_first": a.orthogonalize_first, "adapter_file": a.adapter, "mode": "evaluate"}
    coords = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in a.coords]
    with run.fw.model_variant(cfg):
        pos, neg = run.fw.concept.train_data_fn()
        pos, neg = run.fw._filter_prompts_by_behavior(pos, neg)
        data = PromptData(pos + neg, [True] * len(pos) + [False] * len(neg))
        vecs = run.fw.finder.direction_finder_method.compute_difference_vectors(data, max_positions=max(-p for _, p in coords))
    for layer, p in coords:
        v = vecs[(layer, p)]
        stem = f"{a.out_prefix}-L{layer}-P{p}-direction"
        DirectionVector(vector=v, layer=layer, position_index=p, score=0.0).save(stem)
        print(f"saved {stem} (norm {float(v.norm()):.2f})", flush=True)


if __name__ == "__main__":
    main()
