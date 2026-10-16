"""Concept geometry at every layer, and whether unknown's direction reads familiarity (no generation).

1. For each concept, the difference of means of its training prompts (positive − negative) at the last template
   token, at every layer. Per layer: the cosine between every pair of concepts, and how well each concept's
   direction separates every concept's held-out pairs (AUROC of the projection, positive vs negative). Saved
   directions are found at different layers per concept, so their cosines (RESULTS §17-§24) mix layers; this
   compares like with like.
2. With --familiarity STEM: each prompt's projection onto STEM (unknown's direction) at its layer, for invented
   entities (unknown's eval positives), real but little-known ones (unknown_obscure's) and well-known ones (unknown's
   eval negatives). Is the projection graded by familiarity, and does it predict which little-known questions the
   model says it doesn't know? (--generations: an unknown_obscure evaluate file; its baseline `detected` flags, or
   the judge's INABILITY labels when results/analysis/<file stem>-judge.json exists.)

  .venv/bin/python3 llm-refusal/scripts/concept_geometry.py --model Qwen/Qwen2.5-7B-Instruct \\
      --familiarity results/Qwen2.5-7B-Instruct-unknown-direction \\
      --generations results/obsc-Qwen2.5-7B-Instruct-unknown_obscure-evaluate-Qwen2.5-7B-Instruct-unknown-direction-T128-generations.json

Writes results/analysis/<model>-concept-geometry.json and prints the matrices at a few depths.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from probe import auroc, save_json  # noqa: E402

CONCEPTS = ["refusal", "inability", "realtime", "memory", "persona", "feelings", "unknown", "future"]


def cosine_matrix(diffs: Dict[str, t.Tensor]) -> List[List[float]]:
    """Cosines between the concepts' [d] directions, in dict order."""
    U = t.stack([v.float() / v.float().norm() for v in diffs.values()])
    return (U @ U.T).tolist()


def separation_matrix(diffs: Dict[str, t.Tensor], evals: Dict[str, tuple]) -> List[List[float]]:
    """[direction][eval set]: AUROC of the projection onto each concept's direction, that eval set's positives vs
    negatives (0.5 = no separation; < 0.5 = separates with the opposite sign)."""
    out = []
    for v in diffs.values():
        u = v.float() / v.float().norm()
        out.append([auroc((p.float() @ u).tolist(), (n.float() @ u).tolist()) for p, n in evals.values()])
    return out


def familiarity(proj: Dict[str, List[float]], flags: Optional[List[bool]] = None) -> dict:
    """Mean projection per prompt set, AUROCs between the sets, and (given per-prompt "says it doesn't know" flags
    for the obscure set) the AUROC of the projection for flagged vs unflagged obscure prompts."""
    m = lambda xs: sum(xs) / len(xs)
    out = {"mean": {k: m(v) for k, v in proj.items()},
           "auroc_invented_vs_obscure": auroc(proj["invented"], proj["obscure"]),
           "auroc_obscure_vs_famous": auroc(proj["obscure"], proj["famous"]),
           "auroc_invented_vs_famous": auroc(proj["invented"], proj["famous"])}
    if flags is not None:
        yes = [x for x, f in zip(proj["obscure"], flags) if f]
        no = [x for x, f in zip(proj["obscure"], flags) if not f]
        out.update({"n_obscure_dont_know": len(yes), "auroc_dont_know_within_obscure": auroc(yes, no)})
    return out


def baseline_flags(gens_path: str) -> List[bool]:
    """Per baseline prompt: says it doesn't know. The judge's INABILITY label when its file exists, else the
    detector's flag."""
    stem = os.path.basename(gens_path).removesuffix(".json")
    judge = f"results/analysis/{stem}-judge.json"
    if os.path.exists(judge):
        return [x == "INABILITY" for x in json.load(open(judge))["conditions"]["baseline"]["labels"]]
    return [bool(it["detected"]) for it in json.load(open(gens_path))["baseline"]]


def print_matrix(title: str, names: List[str], cols: List[str], M: List[List[float]]) -> None:
    print(title)
    print(" " * 10 + "".join(f"{c[:8]:>9s}" for c in cols))
    for n, row in zip(names, M):
        print(f"{n[:9]:10s}" + "".join(f"{x:9.2f}" for x in row))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concepts", nargs="+", default=CONCEPTS)
    ap.add_argument("--n", type=int, default=60, help="training prompts per side per concept (the first N)")
    ap.add_argument("--familiarity", metavar="STEM", help="unknown's direction stem, for part 2")
    ap.add_argument("--generations", help="an unknown_obscure evaluate generations file (part 2's per-prompt flags)")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (a smoke check while the GPU is busy)")
    a = ap.parse_args()

    from concept import get_concept
    from datatypes import DirectionVector
    from probe import load_run, residuals_at
    run = load_run(a.model, "refusal", direction=False, **({"force_cpu": True} if a.cpu else {}))
    model, fmt, blocks, short = run.model, run.fmt, run.blocks, run.short
    acts = lambda ps: residuals_at(model, fmt, blocks, ps, -1)          # [n, L, d]

    diffs, evals = {}, {}
    for c in a.concepts:
        cd = get_concept(c)
        tp, tn = cd.train_data_fn()
        diffs[c] = acts(tp[:a.n]).mean(0) - acts(tn[:a.n]).mean(0)      # [L, d]
        ep, en = cd.eval_data_fn()
        evals[c] = (acts(ep[:30]), acts(en[:30]))
        print(f"{c}: {len(tp[:a.n])}+{len(tn[:a.n])} train, {len(ep[:30])}+{len(en[:30])} eval", flush=True)

    n_layers = len(blocks)
    per_layer = []
    for l in range(n_layers):
        dl = {c: v[l] for c, v in diffs.items()}
        el = {c: (p[:, l], n[:, l]) for c, (p, n) in evals.items()}
        per_layer.append({"cos": cosine_matrix(dl), "auroc": separation_matrix(dl, el),
                          "norm": [float(v.norm()) for v in dl.values()]})
    out = {"model": a.model, "concepts": a.concepts, "n_layers": n_layers, "layers": per_layer}
    for f in (0.35, 0.5, 0.65):
        l = int(round(f * (n_layers - 1)))
        print_matrix(f"\n== layer {l}: cosine", a.concepts, a.concepts, per_layer[l]["cos"])
        print_matrix(f"== layer {l}: AUROC (row = direction, column = eval set)", a.concepts, a.concepts,
                     per_layer[l]["auroc"])

    if a.familiarity:
        d = DirectionVector.load(a.familiarity)
        u = d.vector.float().cpu()
        u = u / u.norm()
        inv, fam = get_concept("unknown").eval_data_fn()
        obs, _ = get_concept("unknown_obscure").eval_data_fn()
        proj = {k: (residuals_at(model, fmt, blocks, ps, d.position_index)[:, d.layer] @ u).tolist()
                for k, ps in (("invented", inv), ("obscure", obs), ("famous", fam))}
        flags = baseline_flags(a.generations) if a.generations else None
        if flags is not None:
            prompts = [it["prompt"] for it in json.load(open(a.generations))["baseline"]]
            assert prompts == obs, "the generations file's baseline prompts are not unknown_obscure's positives"
        out["familiarity"] = {"stem": a.familiarity, "layer": d.layer, "projections": proj,
                              **familiarity(proj, flags)}
        print("\n== familiarity:", json.dumps({k: v for k, v in out["familiarity"].items() if k != "projections"},
                                               indent=1))
    path = save_json(f"results/analysis/{short}-concept-geometry.json", out)
    print("saved", path)


if __name__ == "__main__":
    main()
