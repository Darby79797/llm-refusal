"""What direction does a trained rank-one adapter write? (PROPOSED_PLANS item 2)

`--mode rank1` adapters remove refusal through a write direction u that is mostly
orthogonal to the refusal direction r̂, and seeds converge on the same u. This script
takes the seeds' mean direction û (sign-aligned) and:
  1. compares it with every difference-in-means candidate the search scores, i.e.
     every (layer, position), to see whether û is "the refusal direction from
     somewhere else";
  2. ablates it with hooks at all layers (as global ablation does with r̂) and
     reports refusal rate / log-odds on the harmful eval prompts, beside a random
     direction, û's r̂-free part (û leans toward r̂, so ablating û alone also trims
     the r̂ component a little), r̂, and both.

    .venv/bin/python3 llm-refusal/scripts/adapter_direction.py --model meta-llama/Meta-Llama-3-8B-Instruct

Writes results/finetune/<model>-adapter-direction.json.
"""
import argparse
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
_high = os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(min(0.6, 0.75 * float(_high))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from datatypes import DirectionVector, PromptData  # noqa: E402
from framework import DirectionTestFramework  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--tags", default="remove-s0,remove-s1,remove-s2")
    args = ap.parse_args()
    short = args.model.split("/")[-1]

    us = [t.load(f"results/finetune/{short}-{args.concept}-rank1-{tag}-adapters.pt")[0]["U"][:, 0].float()
          for tag in args.tags.split(",")]
    us = [u * t.sign(u @ us[0]) for u in us]          # u v^T = (-u)(-v)^T: align signs
    u_hat = t.stack([u / u.norm() for u in us]).mean(0)
    u_hat = u_hat / u_hat.norm()

    fw = DirectionTestFramework(model_name=args.model, concept=args.concept)
    r = DirectionVector.load(f"results/{short}-{args.concept}-direction")
    r_hat = r.unit.float().cpu()
    cos = lambda a, b: float(t.nn.functional.cosine_similarity(a.float().cpu(), b.float().cpu(), dim=0))
    out = {"model": args.model, "tags": args.tags, "direction": {"layer": r.layer, "position_index": r.position_index},
           "cos_u_rhat": cos(u_hat, r_hat)}

    # 1. û against every difference-in-means candidate (all layers, all searched positions).
    pos, neg = fw.concept.train_data_fn()
    data = PromptData(pos + neg, [True] * len(pos) + [False] * len(neg))
    max_pos = fw.prompt_formatter.assistant_prefix_tokens + 1
    vecs = fw.finder.direction_finder_method.compute_difference_vectors(data, max_positions=max_pos)
    ranked = sorted(((abs(cos(u_hat, v)), l, p, cos(r_hat, v)) for (l, p), v in vecs.items() if v.norm() > 0),
                    reverse=True)
    out["closest_candidates"] = [{"layer": l, "position_index": p, "abs_cos_u": c, "cos_rhat": cr}
                                 for c, l, p, cr in ranked[:8]]
    print("û vs difference-in-means candidates (top 8):")
    for row in out["closest_candidates"]:
        print(f"  L{row['layer']:2d} P{row['position_index']}: |cos(û, cand)| = {row['abs_cos_u']:.3f}  "
              f"(cos(r̂, cand) = {row['cos_rhat']:+.3f})")

    # 2. Ablate û / r̂ / both with hooks at every layer, on the harmful eval prompts.
    eval_pos, _ = fw.concept.eval_data_fn()
    ev, applier = fw.evaluator, fw.intervention_applier
    def run(name, dirs):
        for d in dirs:
            applier.apply_direction_intervention(DirectionVector(vector=d, layer=r.layer, position_index=r.position_index,
                                                                 score=0), "ablate", layers=None)
        try:
            texts = ev.generate_responses(eval_pos, max_new_tokens=64)
            res = {"refusal_rate": sum(map(ev._check_for_detection, texts)) / len(texts),
                   "log_odds": ev._log_odds_metric(eval_pos)}
        finally:
            applier.clear_interventions()
        print(f"  {name:22s} refusal {res['refusal_rate']:.1%}  log-odds {res['log_odds']:+.2f}", flush=True)
        return res
    # Ablating r̂ and û together needs an orthonormal pair (stacked hooks project sequentially).
    u_perp = u_hat - (u_hat @ r_hat) * r_hat
    rand = t.randn(len(u_hat), generator=t.Generator().manual_seed(0))
    out["ablation"] = {"none": run("no intervention", []),
                       "random": run("ablate a random dir.", [rand / rand.norm()]),
                       "u_hat": run("ablate û", [u_hat]),
                       "u_perp": run("ablate û's r̂-free part", [u_perp / u_perp.norm()]),
                       "r_hat": run("ablate r̂", [r_hat]),
                       "both": run("ablate r̂ and û", [r_hat, u_perp / u_perp.norm()])}
    path = f"results/finetune/{short}-adapter-direction.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
