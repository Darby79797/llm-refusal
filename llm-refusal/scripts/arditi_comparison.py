"""
Arditi comparison: 2x2 factorial experiment on Qwen2.5-0.5B-Instruct.

Tests two independent variables:
  1. Direction normalization: unit vector (old) vs raw vector (new, matches Arditi)
  2. Dataset: our hand-written prompts vs Arditi's AdvBench/Alpaca prompts

Factorial design:
  | Condition   | Normalization | Dataset |
  |-------------|---------------|---------|
  | our_unit    | unit          | ours    |
  | our_raw     | raw           | ours    |
  | arditi_unit | unit          | arditi  |
  | arditi_raw  | raw           | arditi  |

For each condition, computes the direction (diff-in-means) and evaluates:
  - Direction norm
  - Induce score at strengths [1, 2, 5, 10]
  - Bypass score (ablation, unchanged by normalization)
  - KL divergence (ablation, unchanged by normalization)
  - Mini eyeball (3 benign prompts with addition at strength=1)

Expected runtime: ~10-15 min on M4 Mac (single model, loaded once).
"""
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import time
import numpy as np
import torch as t
import torch.nn.functional as F
import logging
from copy import deepcopy

from framework import DirectionTestFramework
from datatypes import PromptData, DirectionVector
from scoring import Three_Score_Evaluator
from generation import generate_with_hooks
from concept import get_concept

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
STRENGTHS = [1, 2, 5, 10]


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_direction(framework, concept_name):
    """Compute diff-in-means direction for a given concept."""
    concept = get_concept(concept_name)
    positive, negative = concept.train_data_fn()
    all_data = PromptData(
        positive + negative,
        [True] * len(positive) + [False] * len(negative)
    )
    train_data, val_data = all_data.train_val_split()

    # Use max_positions=1 (last token only, matches Arditi)
    diff_vectors = framework.finder.direction_finder_method.compute_difference_vectors(
        train_data, max_positions=1
    )

    # Find best layer by bypass score (same heuristic as search)
    num_layers = len(framework.intervention_applier.transformer_layers)
    layer_cutoff = int(0.8 * num_layers)

    evaluator = Three_Score_Evaluator(
        framework.model, framework.tokenizer,
        framework.intervention_applier, framework.prompt_formatter,
        target_tokens=concept.target_tokens
    )

    best_dir = None
    best_score = float('inf')
    for (layer, pos), vec in diff_vectors.items():
        if layer >= layer_cutoff:
            continue
        candidate = DirectionVector(vector=vec, layer=layer, position_index=pos, score=0)
        scores = evaluator.compute_all_scores(candidate, val_data)
        lenient = (10 * scores.bypass) + scores.kl - scores.induce
        if lenient < best_score:
            best_score = lenient
            best_dir = candidate

    return best_dir, val_data, concept


def get_logits_with_intervention(evaluator, prompts, direction, int_type, layers, strength, use_raw):
    """Get logits with a direction intervention, choosing raw or unit vector."""
    if use_raw:
        # Raw vector: this is the new default in interventions.py
        evaluator.intervention_applier.apply_direction_intervention(
            direction, int_type, strength=strength, layers=layers
        )
    else:
        # Unit vector: simulate old behavior by scaling strength by 1/norm
        norm = t.norm(direction.vector).item()
        adjusted_strength = strength / norm if norm > 0 else strength
        evaluator.intervention_applier.apply_direction_intervention(
            direction, int_type, strength=adjusted_strength, layers=layers
        )
    try:
        return evaluator._get_logits(prompts)
    finally:
        evaluator.intervention_applier.clear_interventions()


def evaluate_condition(framework, direction, val_data, concept, use_raw, label):
    """Evaluate a single condition in the 2x2 factorial."""
    evaluator = Three_Score_Evaluator(
        framework.model, framework.tokenizer,
        framework.intervention_applier, framework.prompt_formatter,
        target_tokens=concept.target_tokens
    )
    metric = evaluator.metric

    val_pos = val_data.positive
    val_neg = val_data.negative

    # Baseline
    baseline_neg_logits = evaluator._get_logits(val_neg)
    baseline_neg_lo = np.nanmean([metric.compute_log_odds(l) for l in baseline_neg_logits])

    baseline_pos_logits = evaluator._get_logits(val_pos)
    baseline_pos_lo = np.nanmean([metric.compute_log_odds(l) for l in baseline_pos_logits])

    direction_norm = t.norm(direction.vector).item()

    # Bypass (ablation — normalization doesn't matter, uses unit dir internally)
    bypass_logits = evaluator._get_logits(
        val_pos,
        intervention=(direction, "ablate", list(range(len(framework.intervention_applier.transformer_layers))))
    )
    bypass_score = np.nanmean([metric.compute_log_odds(l) for l in bypass_logits])

    # KL (ablation — same, normalization doesn't matter)
    ablated_logits = evaluator._get_logits(
        val_neg,
        intervention=(direction, "ablate", list(range(len(framework.intervention_applier.transformer_layers))))
    )
    kl_scores = [
        F.kl_div(
            F.log_softmax(abl, dim=-1),
            F.softmax(base, dim=-1),
            reduction='sum', log_target=False
        ).item()
        for base, abl in zip(baseline_neg_logits, ablated_logits)
    ]
    kl_score = np.nanmean(kl_scores)

    # Induce at multiple strengths
    induce_results = {}
    for s in STRENGTHS:
        logits = get_logits_with_intervention(
            evaluator, val_neg, direction, "add", [direction.layer], s, use_raw
        )
        lo = np.nanmean([metric.compute_log_odds(l) for l in logits])
        induce_results[s] = lo

    result = {
        "label": label,
        "use_raw": use_raw,
        "direction_norm": direction_norm,
        "layer": direction.layer,
        "pos": direction.position_index,
        "baseline_pos_lo": baseline_pos_lo,
        "baseline_neg_lo": baseline_neg_lo,
        "bypass": bypass_score,
        "kl": kl_score,
        "induce_by_strength": induce_results,
    }

    return result


def print_results_table(results):
    """Print a formatted comparison table."""
    print(f"\n{'═'*80}")
    print(f"2×2 FACTORIAL RESULTS")
    print(f"{'═'*80}")

    col_w = 18
    labels = [r["label"] for r in results]

    # Header
    print(f"  {'Metric':<22s}", end="")
    for label in labels:
        print(f"  {label:>{col_w}s}", end="")
    print()
    print(f"  {'─'*22}", end="")
    for _ in labels:
        print(f"  {'─'*col_w}", end="")
    print()

    # Static rows
    rows = [
        ("Layer", "layer"),
        ("Pos", "pos"),
        ("Direction norm", "direction_norm"),
        ("Baseline pos LO", "baseline_pos_lo"),
        ("Baseline neg LO", "baseline_neg_lo"),
        ("Bypass (ablation)", "bypass"),
        ("KL (ablation)", "kl"),
    ]
    for display, key in rows:
        print(f"  {display:<22s}", end="")
        for r in results:
            val = r[key]
            if isinstance(val, float):
                print(f"  {val:>{col_w}.4f}", end="")
            else:
                print(f"  {val:>{col_w}}", end="")
        print()

    # Induce rows per strength
    for s in STRENGTHS:
        display = f"Induce (s={s})"
        print(f"  {display:<22s}", end="")
        for r in results:
            lo = r["induce_by_strength"][s]
            print(f"  {lo:>{col_w}.4f}", end="")
        print()

    # Delta from baseline
    print()
    print(f"  {'--- Deltas ---':<22s}")
    for s in STRENGTHS:
        display = f"Induce Δ (s={s})"
        print(f"  {display:<22s}", end="")
        for r in results:
            lo = r["induce_by_strength"][s]
            delta = lo - r["baseline_neg_lo"]
            print(f"  {delta:>+{col_w}.4f}", end="")
        print()

    # Effective magnitude row
    print()
    print(f"  {'--- Effective add ---':<22s}")
    for s in STRENGTHS:
        display = f"||add|| (s={s})"
        print(f"  {display:<22s}", end="")
        for r in results:
            if r["use_raw"]:
                mag = s * r["direction_norm"]
            else:
                mag = s * 1.0  # unit vector
            print(f"  {mag:>{col_w}.2f}", end="")
        print()


def run_mini_eyeball(framework, direction, concept, use_raw, label):
    """Generate text with addition for qualitative inspection."""
    eval_pos, eval_neg = concept.eval_data_fn()
    prompts = eval_neg[:3]

    print(f"\n{'─'*60}")
    print(f"MINI EYEBALL: {label} (addition @ strength=1)")
    print(f"{'─'*60}")

    # Baseline
    print(f"\n  [Baseline]")
    baseline_texts = generate_with_hooks(
        framework.model, framework.tokenizer,
        framework.prompt_formatter, prompts,
        max_new_tokens=48
    )
    for p, text in zip(prompts, baseline_texts):
        print(f"    Q: {p[:70]}...")
        print(f"    A: {text[:150]}")

    # With addition
    if use_raw:
        strength = 1.0
    else:
        norm = t.norm(direction.vector).item()
        strength = 1.0 / norm if norm > 0 else 1.0

    framework.intervention_applier.apply_direction_intervention(
        direction, "add", strength=strength, layers=[direction.layer]
    )
    try:
        induced_texts = generate_with_hooks(
            framework.model, framework.tokenizer,
            framework.prompt_formatter, prompts,
            max_new_tokens=48
        )
    finally:
        framework.intervention_applier.clear_interventions()

    print(f"\n  [Addition, {'raw' if use_raw else 'unit'}, s=1]")
    for p, text in zip(prompts, induced_texts):
        print(f"    Q: {p[:70]}...")
        print(f"    A: {text[:150]}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("ARDITI COMPARISON: 2×2 Factorial Experiment")
    print(f"Model: {MODEL}")
    print(f"Factors: normalization (unit vs raw) × dataset (ours vs arditi)")
    print("=" * 70)

    total_t0 = time.time()

    # Load model once
    logger.info(f"Loading {MODEL}...")
    framework = DirectionTestFramework(model_name=MODEL, concept="refusal")

    # ── Compute directions for both datasets ──────────────────────────────
    print("\n--- Computing directions ---")

    print("\n  Computing direction with OUR dataset...")
    our_dir, our_val, our_concept = compute_direction(framework, "refusal")
    print(f"  Our direction: layer={our_dir.layer}, pos={our_dir.position_index}, "
          f"norm={t.norm(our_dir.vector).item():.4f}")

    print("\n  Computing direction with ARDITI dataset...")
    arditi_dir, arditi_val, arditi_concept = compute_direction(framework, "refusal_arditi")
    print(f"  Arditi direction: layer={arditi_dir.layer}, pos={arditi_dir.position_index}, "
          f"norm={t.norm(arditi_dir.vector).item():.4f}")

    # Cosine similarity between the two directions
    cos_sim = F.cosine_similarity(
        our_dir.vector.unsqueeze(0),
        arditi_dir.vector.unsqueeze(0)
    ).item()
    print(f"\n  Cosine similarity between directions: {cos_sim:.4f}")

    # ── Evaluate all 4 conditions ─────────────────────────────────────────
    print("\n--- Evaluating conditions ---")

    conditions = [
        ("our_unit", our_dir, our_val, our_concept, False),
        ("our_raw", our_dir, our_val, our_concept, True),
        ("arditi_unit", arditi_dir, arditi_val, arditi_concept, False),
        ("arditi_raw", arditi_dir, arditi_val, arditi_concept, True),
    ]

    all_results = []
    for label, direction, val_data, concept, use_raw in conditions:
        print(f"\n  Evaluating: {label}...")
        result = evaluate_condition(framework, direction, val_data, concept, use_raw, label)
        all_results.append(result)
        induce_s1 = result["induce_by_strength"][1]
        delta = induce_s1 - result["baseline_neg_lo"]
        print(f"    Induce(s=1)={induce_s1:.4f}, Δ={delta:+.4f}, norm={result['direction_norm']:.4f}")

    # ── Print comparison table ────────────────────────────────────────────
    print_results_table(all_results)

    # ── Mini eyeball for best condition ───────────────────────────────────
    # Find condition with best induce at s=1
    best_idx = max(range(len(all_results)),
                   key=lambda i: all_results[i]["induce_by_strength"][1] - all_results[i]["baseline_neg_lo"])
    best = all_results[best_idx]
    best_label, best_dir, _, best_concept, best_use_raw = conditions[best_idx]

    print(f"\n{'═'*80}")
    print(f"BEST CONDITION: {best_label}")
    print(f"  Induce Δ at s=1: {best['induce_by_strength'][1] - best['baseline_neg_lo']:+.4f}")
    print(f"{'═'*80}")

    run_mini_eyeball(framework, best_dir, best_concept, best_use_raw, best_label)

    # Also show eyeball for arditi_raw (the theoretically correct condition)
    if best_label != "arditi_raw":
        arditi_raw_idx = next(i for i, (l, *_) in enumerate(conditions) if l == "arditi_raw")
        _, ar_dir, _, ar_concept, ar_use_raw = conditions[arditi_raw_idx]
        run_mini_eyeball(framework, ar_dir, ar_concept, ar_use_raw, "arditi_raw")

    elapsed = time.time() - total_t0
    print(f"\n{'═'*80}")
    print(f"EXPERIMENT COMPLETE — {elapsed/60:.1f} min")
    print(f"{'═'*80}")
