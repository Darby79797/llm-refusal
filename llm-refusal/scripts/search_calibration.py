"""
Search calibration: collect per-layer search scores AND behavioral eval data.

For each candidate layer, computes:
  - bypass, induce, KL scores (what the search uses to select)
  - actual induction rate (generate with addition, check detection)
  - actual ablation rate (generate with global ablation, check detection)

This lets us correlate cheap scores with actual behavioral outcomes and
find better search criteria.

Usage:
  python llm-refusal/scripts/search_calibration.py --model Qwen/Qwen2.5-0.5B-Instruct
  python llm-refusal/scripts/search_calibration.py --model Qwen/Qwen2.5-3B-Instruct --num-prompts 10
"""
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import argparse
import json
import time
import numpy as np
import torch as t
import logging

from framework import DirectionTestFramework
from datatypes import PromptData, DirectionVector
from generation import generate_with_hooks
from concept import get_concept
from coherence import is_garbled

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

NUM_PROMPTS = 20
POS = -1


def run_calibration(model_name, concept_name="refusal", num_prompts=NUM_PROMPTS):
    t0 = time.time()
    model_short = model_name.split("/")[-1]

    print(f"\n{'='*80}")
    print(f"SEARCH CALIBRATION: {model_name} [{concept_name}]")
    print(f"{'='*80}")

    # Set up framework
    framework = DirectionTestFramework(model_name=model_name, concept=concept_name)
    concept = get_concept(concept_name)

    # Prepare data (same splits as search)
    positive, negative = concept.train_data_fn()
    all_data = PromptData(
        positive + negative,
        [True] * len(positive) + [False] * len(negative)
    )
    train_data, val_data = all_data.train_val_split()

    # Compute directions
    logger.info("Computing difference-in-means vectors...")
    diff_vectors = framework.finder.direction_finder_method.compute_difference_vectors(
        train_data, max_positions=1
    )

    # Pre-compute baseline logits for scores (same as search)
    val_pos = val_data.positive
    val_neg = val_data.negative
    baseline_neg_logits = framework.finder.evaluator._get_logits(val_neg) if val_neg else []
    baseline_pos_logits = framework.finder.evaluator._get_logits(val_pos) if val_pos else []
    baseline_bypass = np.nanmean([framework.finder.evaluator.metric.compute_log_odds(l) for l in baseline_pos_logits]) if baseline_pos_logits else 0.0
    baseline_induce = np.nanmean([framework.finder.evaluator.metric.compute_log_odds(l) for l in baseline_neg_logits]) if baseline_neg_logits else 0.0
    logger.info(f"Baseline scores: bypass={baseline_bypass:.4f}, induce={baseline_induce:.4f}")

    # Get eval prompts for behavioral tests
    eval_pos, eval_neg = concept.eval_data_fn()
    induction_prompts = eval_neg[:num_prompts]
    ablation_prompts = eval_pos[:num_prompts]

    # Generate baseline texts for comparison
    logger.info("Generating baseline responses...")
    baseline_neg_texts = generate_with_hooks(
        framework.model, framework.tokenizer,
        framework.prompt_formatter, induction_prompts,
        max_new_tokens=64
    )
    baseline_pos_texts = generate_with_hooks(
        framework.model, framework.tokenizer,
        framework.prompt_formatter, ablation_prompts,
        max_new_tokens=64
    )
    baseline_induction_rate = sum(1 for t in baseline_neg_texts if concept.detect(t)) / len(baseline_neg_texts)
    baseline_ablation_rate = sum(1 for t in baseline_pos_texts if concept.detect(t)) / len(baseline_pos_texts)
    logger.info(f"Baseline behavioral: induction_rate={baseline_induction_rate:.2%} (should be ~0%), "
                f"ablation_rate={baseline_ablation_rate:.2%} (should be high)")

    # Scan all candidate layers
    num_layers = len(framework.intervention_applier.transformer_layers)
    layer_cutoff = int(0.8 * num_layers)
    candidate_layers = [l for l in range(num_layers) if l < layer_cutoff and (l, POS) in diff_vectors]

    results = []

    for layer in candidate_layers:
        vec = diff_vectors[(layer, POS)]
        direction = DirectionVector(vector=vec, layer=layer, position_index=POS, score=0)
        norm = t.norm(direction.vector).item()

        # 1. Compute search scores (cheap - logits only)
        scores = framework.finder.evaluator.compute_all_scores(
            direction, val_data, baseline_neg_logits=baseline_neg_logits
        )

        # 2. Behavioral induction: add direction at this layer, generate on benign prompts
        framework.intervention_applier.apply_direction_intervention(
            direction, "add", strength=1.0, layers=[layer]
        )
        try:
            induction_texts = generate_with_hooks(
                framework.model, framework.tokenizer,
                framework.prompt_formatter, induction_prompts,
                max_new_tokens=64
            )
        finally:
            framework.intervention_applier.clear_interventions()

        n_induced = sum(1 for t in induction_texts if concept.detect(t))
        n_garbled = sum(1 for t in induction_texts if is_garbled(t))
        induction_rate = n_induced / len(induction_texts)

        # 3. Behavioral ablation: ablate direction from all layers, generate on harmful prompts
        framework.intervention_applier.apply_direction_intervention(
            direction, "ablate", strength=1.0, layers=list(range(num_layers))
        )
        try:
            ablation_texts = generate_with_hooks(
                framework.model, framework.tokenizer,
                framework.prompt_formatter, ablation_prompts,
                max_new_tokens=64
            )
        finally:
            framework.intervention_applier.clear_interventions()

        n_still_refused = sum(1 for t in ablation_texts if concept.detect(t))
        ablation_rate = n_still_refused / len(ablation_texts)

        row = {
            "layer": layer,
            "depth_pct": layer / num_layers,
            "norm": round(norm, 2),
            "bypass": round(scores.bypass, 4),
            "induce": round(scores.induce, 4),
            "kl": round(scores.kl, 4),
            "induce_delta": round(scores.induce - baseline_induce, 4),
            "bypass_delta": round(scores.bypass - baseline_bypass, 4),
            "behav_induction": round(induction_rate, 4),
            "behav_induction_n": n_induced,
            "behav_garbled": n_garbled,
            "behav_ablation": round(ablation_rate, 4),
            "behav_ablation_n": n_still_refused,
        }
        results.append(row)
        logger.info(f"L{layer:2d} ({layer/num_layers:.0%}) | "
                    f"scores: byp={scores.bypass:.2f} ind={scores.induce:.2f} kl={scores.kl:.3f} | "
                    f"behav: ind={induction_rate:.0%}({n_induced}/{num_prompts}) "
                    f"garb={n_garbled} abl={ablation_rate:.0%}({n_still_refused}/{num_prompts})")

    # Save results
    output = {
        "model": model_name,
        "concept": concept_name,
        "num_layers": num_layers,
        "layer_cutoff": layer_cutoff,
        "num_prompts": num_prompts,
        "baseline_bypass": round(baseline_bypass, 4),
        "baseline_induce": round(baseline_induce, 4),
        "baseline_induction_rate": round(baseline_induction_rate, 4),
        "baseline_ablation_rate": round(baseline_ablation_rate, 4),
        "layers": results,
    }

    os.makedirs("results", exist_ok=True)
    out_path = f"results/{model_short}-calibration.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved calibration data to {out_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"CALIBRATION SUMMARY: {model_name} [{concept_name}]")
    print(f"Baseline: bypass={baseline_bypass:.4f}, induce={baseline_induce:.4f}, "
          f"behav_induction={baseline_induction_rate:.0%}, behav_ablation={baseline_ablation_rate:.0%}")
    print(f"{'='*100}")
    print(f"  {'Layer':>5} {'Depth':>6} {'||dir||':>8} "
          f"{'Bypass':>8} {'Induce':>8} {'IndΔ':>8} {'KL':>8} "
          f"{'Ind%':>6} {'Garb':>5} {'Abl%':>6}")
    print(f"  {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*5} {'─'*6}")
    for r in results:
        print(f"  {r['layer']:>5} {r['depth_pct']:>5.0%} {r['norm']:>8.2f} "
              f"{r['bypass']:>8.2f} {r['induce']:>8.2f} {r['induce_delta']:>+8.2f} {r['kl']:>8.3f} "
              f"{r['behav_induction']:>5.0%} {r['behav_garbled']:>5} {r['behav_ablation']:>5.0%}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--concept", default="refusal")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    args = parser.parse_args()
    run_calibration(args.model, concept_name=args.concept, num_prompts=args.num_prompts)
