"""Do regrowth seeds learn the same refusal direction? Same-layer comparison across adapters.

The full-depth searches pick different layers per seed (Llama-3-8B L16 vs L29, Qwen-7B L24 vs L22), and comparing
the winners mixes the seed effect with depth drift. This computes the harmful-minus-harmless difference-in-means at
one template position at every layer inside each model variant (clean; edited; edited + each adapter) and reports
the cosine between variants layer by layer. Train prompts, unfiltered (the search filters by behaviour first), so
the vectors approximate rather than reproduce the search's candidates.

  .venv/bin/python3 llm-refusal/scripts/regrown_seed_directions.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --arms readers-r32,readers-r32-s1,readers-r32-s2,readers-r0

Writes results/analysis/<short>-regrown-seed-directions.json (cosines) and .pt (the vectors); resumes per variant.
"""
import argparse
import contextlib
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from finetune import installed  # noqa: E402
from orthogonalize import orthogonalized  # noqa: E402
from probe import cos, load_run, residuals_at, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", required=True, help="comma-separated regrow tags (results/finetune/<short>-refusal-regrow-<tag>)")
    ap.add_argument("--pos", type=int, default=-1)
    ap.add_argument("--n", type=int, default=128, help="prompts per class")
    a = ap.parse_args()

    run = load_run(a.model)
    model = run.model
    pos, neg = run.fw.concept.train_data_fn()
    pos, neg = pos[:a.n], neg[:a.n]
    json_path = run.path("analysis", "regrown-seed-directions")
    pt_path = json_path[:-5] + ".pt"
    vecs = t.load(pt_path) if os.path.exists(pt_path) else {}
    if vecs:
        print("resuming; done:", list(vecs), flush=True)

    def diff_means():
        hp = residuals_at(model, run.fmt, run.blocks, pos, a.pos).double().mean(0)
        hn = residuals_at(model, run.fmt, run.blocks, neg, a.pos).double().mean(0)
        return (hp - hn).float()  # [n_layers, d_model]

    def record(name, ctx=contextlib.nullcontext):
        if name in vecs:
            return
        with ctx():
            vecs[name] = diff_means()
        t.save(vecs, pt_path + ".tmp")
        os.replace(pt_path + ".tmp", pt_path)
        print("done", name, flush=True)

    record("clean")
    with orthogonalized(model, run.r.vector):
        record("edited")
        for arm in a.arms.split(","):
            stem = f"results/finetune/{run.short}-refusal-regrow-{arm}"
            if not os.path.exists(stem + ".json"):
                print(f"skip {arm}: no {stem}.json", flush=True)
                continue
            record(arm, lambda stem=stem: installed(model, stem, "full"))

    names = list(vecs)
    n_layers = next(iter(vecs.values())).shape[0]
    out = {"model": a.model, "pos": a.pos, "n_per_class": len(pos), "variants": names, "by_layer": []}
    for layer in range(n_layers):
        row = {"layer": layer, "norm": {k: float(vecs[k][layer].norm()) for k in names},
               "cos_rhat": {k: cos(vecs[k][layer], run.r_hat) for k in names},
               "cos": {f"{x}|{y}": cos(vecs[x][layer], vecs[y][layer]) for x, y in itertools.combinations(names, 2)}}
        out["by_layer"].append(row)
    regrown = [k for k in names if k not in ("clean", "edited") and k.split("-")[1] != "r0"]
    # What refusal training added: each regrown seed minus its benign-only twin (same arm and seed, r0).
    twin = {k: "-".join([k.split("-")[0], "r0", *k.split("-")[2:]]) for k in regrown}
    deltas = {k: vecs[k] - vecs[twin[k]] for k in regrown if twin[k] in vecs}
    out["regrowth_delta_twins"] = {k: twin[k] for k in deltas}
    for row in out["by_layer"]:
        L = row["layer"]
        row["delta_norm"] = {k: float(d[L].norm()) for k, d in deltas.items()}
        row["delta_cos"] = {f"{x}|{y}": cos(deltas[x][L], deltas[y][L]) for x, y in itertools.combinations(deltas, 2)}
        row["delta_cos_rhat"] = {k: cos(d[L], run.r_hat) for k, d in deltas.items()}
    print(f"by layer at P{a.pos}: cos between regrown seeds (raw diff-means) || cos between regrowth deltas (minus the "
          f"benign twin) || cos(edited, seed) for scale", flush=True)
    for row in out["by_layer"]:
        raw = " ".join(f"{v:+.2f}" for p, v in row["cos"].items() if all(s in regrown for s in p.split("|")))
        dl = " ".join(f"{v:+.2f}" for v in row["delta_cos"].values())
        ed = " ".join(f"{row['cos'][f'edited|{k}']:+.2f}" for k in regrown if f"edited|{k}" in row["cos"])
        print(f"  L{row['layer']:2d} {raw} || {dl} || {ed}", flush=True)
    print("saved", save_json(json_path, out))


if __name__ == "__main__":
    main()
