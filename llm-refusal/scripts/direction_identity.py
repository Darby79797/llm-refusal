"""Are the two r̂-free directions fine-tuning finds the same thing? (limb 2.1)

The rank1 remove adapter writes an inhibitor û⊥ (orthogonal to r̂) that switches refusal
off. The regrow runs rebuild refusal after the weight edit along a direction with cos 0
to r̂ (readers arm). Both adapter sets are on disk. For each regrow arm, inside the
weight edit with the LoRA installed, this re-extracts the harmful-harmless
difference-in-means at r̂'s coordinates and reports:

  cos with r̂, with û (rank1 seeds' mean) and with û⊥; cos between arms
  in the r32 arms (refusal regrown): refusal on 50 harmful eval prompts with nothing /
  the regrown direction ablated at every layer / û⊥ ablated / a random direction ablated
  AUROC of û⊥ as a harmful-vs-harmless feature in the regrown model

  .venv/bin/python3 llm-refusal/scripts/direction_identity.py --model Qwen/Qwen2.5-0.5B-Instruct --rank1-tags remove

Writes results/finetune/<model>-direction-identity.json.
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from datatypes import DirectionVector, PromptData  # noqa: E402
from finetune import installed, load_adapters  # noqa: E402
from framework import DirectionTestFramework  # noqa: E402
from orthogonalize import edit_bytes, orthogonalized  # noqa: E402
from probe import auroc, behaviour, cos, residuals_at, unit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank1-tags", default="remove-s0,remove-s1,remove-s2")
    ap.add_argument("--arms", default="writers-r0,writers-r32,readers-r0,readers-r32")
    ap.add_argument("--n-eval", type=int, default=50)
    a = ap.parse_args()
    short = a.model.split("/")[-1]
    fw = DirectionTestFramework(model_name=a.model, concept="refusal")
    model, fmt, applier = fw.model, fw.prompt_formatter, fw.intervention_applier
    blocks = applier.transformer_layers
    r = DirectionVector.load(f"results/{short}-refusal-direction")
    r_hat = r.unit.float().cpu()

    us = [load_adapters(f"results/finetune/{short}-refusal-rank1-{tag}")[1][0]["U"][:, 0].float() for tag in a.rank1_tags.split(",")]
    us = [u * t.sign(u @ us[0]) for u in us]
    u_hat = unit(t.stack([unit(u) for u in us]).mean(0))
    u_perp = unit(u_hat - (u_hat @ r_hat) * r_hat)
    rand = unit(t.randn(len(r_hat), generator=t.Generator().manual_seed(0)))

    train_pos, train_neg = fw.concept.train_data_fn()
    eval_pos, eval_neg = fw.concept.eval_data_fn()
    eval_pos, eval_neg = eval_pos[:a.n_eval], eval_neg[:a.n_eval]
    fw.evaluator.reserve_bytes = edit_bytes(model)

    out = {"model": a.model, "direction": {"layer": r.layer, "position_index": r.position_index},
           "rank1_tags": a.rank1_tags, "cos_u_hat_r_hat": cos(u_hat, r_hat), "arms": {}}
    regrown = {}
    for arm in a.arms.split(","):
        stem = f"results/finetune/{short}-refusal-regrow-{arm}"
        if not os.path.exists(stem + ".json"):
            print(f"skip {arm}: no {stem}.json")
            continue
        with orthogonalized(model, r.vector), installed(model, stem, "full"):
            data = PromptData(train_pos + train_neg, [True] * len(train_pos) + [False] * len(train_neg))
            vecs = fw.finder.direction_finder_method.compute_difference_vectors(data, max_positions=-r.position_index)
            d = vecs[(r.layer, r.position_index)].float()
            regrown[arm] = d
            res = {"norm": float(d.norm()), "clean_norm": float(r.vector.norm()),
                   "cos_r_hat": cos(d, r_hat), "cos_u_hat": cos(d, u_hat), "cos_u_perp": cos(d, u_perp),
                   "cos_random": cos(d, rand)}
            # Per-layer: where does the regrown contrast align with û⊥ most?
            res["cos_u_perp_by_layer"] = [cos(vecs[(l, r.position_index)], u_perp) if (l, r.position_index) in vecs else None
                                          for l in range(len(blocks))]
            # û⊥ as a feature in this model.
            hp = residuals_at(model, fmt, blocks, eval_pos, r.position_index)[:, r.layer] @ u_perp
            hn = residuals_at(model, fmt, blocks, eval_neg, r.position_index)[:, r.layer] @ u_perp
            res["u_perp_feature_auroc"] = auroc(hp, hn)
            hp = residuals_at(model, fmt, blocks, eval_pos, r.position_index)[:, r.layer] @ unit(d)
            hn = residuals_at(model, fmt, blocks, eval_neg, r.position_index)[:, r.layer] @ unit(d)
            res["regrown_feature_auroc"] = auroc(hp, hn)
            if arm.endswith("r32"):
                abl = lambda v: (lambda: applier.intervened(v, "ablate", layers=None))  # noqa: E731
                res["refusal"] = {"none": behaviour(fw, eval_pos, contextlib.nullcontext),
                                  "ablate_regrown": behaviour(fw, eval_pos, abl(unit(d))),
                                  "ablate_u_perp": behaviour(fw, eval_pos, abl(u_perp)),
                                  "ablate_random": behaviour(fw, eval_pos, abl(rand)),
                                  "harmless_none": behaviour(fw, eval_neg, contextlib.nullcontext),
                                  "harmless_add_regrown": behaviour(fw, eval_neg, lambda: applier.intervened(
                                      DirectionVector(vector=d, layer=r.layer, position_index=r.position_index, score=0),
                                      "add", layers=[r.layer]))}
        out["arms"][arm] = res
        print(f"{arm:14s} norm {res['norm']:.2f} (clean {res['clean_norm']:.2f})  cos: r̂ {res['cos_r_hat']:+.3f}  "
              f"û {res['cos_u_hat']:+.3f}  û⊥ {res['cos_u_perp']:+.3f}  rand {res['cos_random']:+.3f} | "
              f"AUROC û⊥ {res['u_perp_feature_auroc']:.2f} regrown {res['regrown_feature_auroc']:.2f}"
              + (" | refusal " + " ".join(f"{k}={v['rate']:.0%}" for k, v in res["refusal"].items()) if "refusal" in res else ""),
              flush=True)
    out["cos_between_arms"] = {f"{x}|{y}": cos(regrown[x], regrown[y]) for x in regrown for y in regrown if x < y}
    print("between arms:", {k: round(v, 3) for k, v in out["cos_between_arms"].items()})
    path = f"results/finetune/{short}-direction-identity.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
