"""Is the regrown model's refusal axis the clean model's own downstream contrast? (referee test)

The search inside the regrown Llama-3-8B model selects L16/P-1, and the clean model's
search CSV already scores its own L16/P-1 difference-in-means as "induces strongly, not
necessary". If the regrown axis is simply that clean contrast with r̂ projected out, then
"regrowth rebuilds a new axis" is really "the edit left the clean model's downstream
refusal representation in place and regrowth re-routes through it". This computes the
clean difference-in-means at the regrown axis's coordinates (train prompts), projects
r̂ out of it, and reports cosines with the regrown axis, plus the same at every layer at
that position.

  .venv/bin/python3 llm-refusal/scripts/axis_vs_clean.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --axis results/regrown-Meta-Llama-3-8B-Instruct-refusal-direction

Writes results/analysis/<short>-axis-vs-clean.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

from datatypes import DirectionVector, PromptData  # noqa: E402
from probe import cos, load_run, save_json, unit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--axis", required=True, help="direction stem of the regrown model's selected direction")
    a = ap.parse_args()
    run = load_run(a.model)
    axis = DirectionVector.load(a.axis)
    ax = unit(axis.vector)
    pos, neg = run.fw.concept.train_data_fn()
    data = PromptData(pos + neg, [True] * len(pos) + [False] * len(neg))
    vecs = run.fw.finder.direction_finder_method.compute_difference_vectors(data, max_positions=-axis.position_index)
    out = {"model": a.model, "axis": {"layer": axis.layer, "position_index": axis.position_index},
           "r_hat": run.coords, "by_layer": []}
    for layer in range(len(run.blocks)):
        v = vecs.get((layer, axis.position_index))
        if v is None:
            continue
        v = v.float()
        v_perp = v - (v @ run.r_hat) * run.r_hat
        row = {"layer": layer, "cos_clean_axis": cos(v, ax), "cos_clean_perp_axis": cos(v_perp, ax),
               "cos_clean_rhat": cos(v, run.r_hat), "norm": float(v.norm())}
        out["by_layer"].append(row)
    at = next(r for r in out["by_layer"] if r["layer"] == axis.layer)
    out["at_axis_layer"] = at
    print(f"clean difference-in-means at L{axis.layer}/P{axis.position_index}: cos with regrown axis {at['cos_clean_axis']:+.3f}, "
          f"with r̂ projected out {at['cos_clean_perp_axis']:+.3f}; cos(clean, r̂) {at['cos_clean_rhat']:+.3f}", flush=True)
    print("by layer (cos clean⊥ vs axis): " + " ".join(f"L{r['layer']}:{r['cos_clean_perp_axis']:+.2f}" for r in out["by_layer"]), flush=True)
    print("saved", save_json(run.path("analysis", "axis-vs-clean"), out))


if __name__ == "__main__":
    main()
