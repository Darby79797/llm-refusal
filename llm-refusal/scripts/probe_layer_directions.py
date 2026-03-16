"""
Probe refusal direction vectors at layers 12, 13, and 15 (pos -1) on Qwen1.5-1.8B-Chat.

Layer 12 = search winner, Layer 13 = prior manual testing, Layer 15 = Arditi et al. paper.
Runs fast first-token metrics, top-token analysis, mini eyeball generation,
and strength/multi-layer follow-ups if induction fails.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch as t
import torch.nn.functional as F
import logging

from framework import DirectionTestFramework
from datatypes import PromptData, DirectionVector
from scoring import Three_Score_Evaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
PROBE_LAYERS = [12, 13, 15]
POS = -1


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_logits_custom(evaluator, prompts, direction, int_type, layers, strength):
    """Get logits with a custom intervention strength (bypasses _get_logits' hardcoded strength=1.0)."""
    evaluator.intervention_applier.apply_direction_intervention(
        direction, int_type, strength=strength, layers=layers
    )
    try:
        return evaluator._get_logits(prompts)
    finally:
        evaluator.intervention_applier.clear_interventions()


def show_top_tokens(evaluator, prompts, labels, intervention=None, title="", n=10):
    """
    Show top-n predicted next tokens with probabilities.
    intervention: (direction, int_type, layers, strength) or None for baseline.
    """
    if intervention:
        direction, int_type, layers, strength = intervention
        evaluator.intervention_applier.apply_direction_intervention(
            direction, int_type, strength=strength, layers=layers
        )
    try:
        logits_list = evaluator._get_logits(prompts)
    finally:
        evaluator.intervention_applier.clear_interventions()

    print(f"\n{'─'*60}")
    print(f"TOP-{n} TOKENS: {title}")
    print(f"{'─'*60}")
    for i, (prompt, logits) in enumerate(zip(prompts, logits_list)):
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = t.topk(probs, n)
        tokens = [evaluator.tokenizer.decode([tid]) for tid in top_ids]
        log_odds = evaluator.metric.compute_log_odds(logits)

        print(f"\n  [{labels[i]}] {prompt[:80]}")
        print(f"  Log-odds(refusal): {log_odds:.4f}")
        for rank, (tok, p) in enumerate(zip(tokens, top_probs)):
            bar = "█" * int(p.item() * 50)
            print(f"    {rank+1:2d}. {tok!r:15s} {p.item():.4f} {bar}")


def print_scores_table(all_results):
    """Print a side-by-side comparison table of scores across layers."""
    layers = sorted(all_results.keys())

    print(f"\n{'═'*70}")
    print(f"THREE-SCORE COMPARISON (pos={POS})")
    print(f"{'═'*70}")
    print(f"  {'Metric':<28s}", end="")
    for layer in layers:
        print(f"  {'Layer '+str(layer):>10s}", end="")
    print()
    print(f"  {'─'*28}", end="")
    for _ in layers:
        print(f"  {'─'*10}", end="")
    print()

    rows = [
        ("Bypass (lower=better)",    "bypass"),
        ("Induce (higher=better)",   "induce"),
        ("KL div (lower=better)",    "kl"),
        ("Baseline pos log-odds",    "baseline_pos"),
        ("Baseline neg log-odds",    "baseline_neg"),
        ("Subtract pos log-odds",    "subtract_pos"),
        ("Subtract neg log-odds",    "subtract_neg"),
    ]
    for display, key in rows:
        if key in all_results[layers[0]]:
            print(f"  {display:<28s}", end="")
            for layer in layers:
                print(f"  {all_results[layer][key]:>10.4f}", end="")
            print()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("REFUSAL DIRECTION PROBE: Layers 12, 13, 15 (pos -1)")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    # ── Setup ────────────────────────────────────────────────────────────
    framework = DirectionTestFramework(model_name=MODEL_NAME, concept="refusal")
    concept = framework.concept

    evaluator = Three_Score_Evaluator(
        framework.model, framework.tokenizer,
        framework.intervention_applier, framework.prompt_formatter,
        target_tokens=concept.target_tokens
    )
    metric = evaluator.metric
    num_layers = len(framework.intervention_applier.transformer_layers)

    # ── Data (matching search split: 20%, random_state=39) ───────────────
    positive, negative = concept.train_data_fn()
    all_data = PromptData(
        positive + negative,
        [True] * len(positive) + [False] * len(negative)
    )
    train_data, val_data = all_data.train_val_split()

    val_pos = [p for p, l in zip(val_data.prompts, val_data.labels) if l]
    val_neg = [p for p, l in zip(val_data.prompts, val_data.labels) if not l]
    print(f"\nData: {len(train_data.prompts)} train, {len(val_data.prompts)} val "
          f"({len(val_pos)} pos, {len(val_neg)} neg)")

    eval_pos, eval_neg = concept.eval_data_fn()

    # ── Compute difference-in-means (one pass, all layers) ───────────────
    print("\nComputing difference-in-means vectors...")
    diff_vectors = framework.finder.direction_finder_method.compute_difference_vectors(train_data)

    directions = {}
    for layer in PROBE_LAYERS:
        vec = diff_vectors.get((layer, POS))
        if vec is None:
            print(f"  WARNING: No vector at ({layer}, {POS})")
            continue
        directions[layer] = DirectionVector(vector=vec, layer=layer, position_index=POS, score=0.0)
        print(f"  Layer {layer}: vector norm = {t.norm(vec).item():.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Three-Score Comparison
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Three-Score Comparison")
    print("=" * 70)

    # Baseline logits (computed once, shared across layers)
    print("Computing baseline logits...")
    baseline_pos_logits = evaluator._get_logits(val_pos)
    baseline_neg_logits = evaluator._get_logits(val_neg)

    baseline_pos_lo = np.nanmean([metric.compute_log_odds(l) for l in baseline_pos_logits])
    baseline_neg_lo = np.nanmean([metric.compute_log_odds(l) for l in baseline_neg_logits])
    print(f"  Baseline pos log-odds: {baseline_pos_lo:.4f}  (sanity: should be ~ -9.32)")
    print(f"  Baseline neg log-odds: {baseline_neg_lo:.4f}  (sanity: should be ~ -11.54)")

    results = {}
    for layer, direction in directions.items():
        print(f"\nScoring layer {layer}...")
        scores = evaluator.compute_all_scores(direction, val_data, baseline_neg_logits=baseline_neg_logits)

        # Subtract intervention (layer-specific)
        sub_pos_logits = get_logits_custom(evaluator, val_pos, direction, "subtract", [layer], 1.0)
        sub_neg_logits = get_logits_custom(evaluator, val_neg, direction, "subtract", [layer], 1.0)

        results[layer] = {
            "bypass": scores.bypass,
            "induce": scores.induce,
            "kl": scores.kl,
            "baseline_pos": baseline_pos_lo,
            "baseline_neg": baseline_neg_lo,
            "subtract_pos": np.nanmean([metric.compute_log_odds(l) for l in sub_pos_logits]),
            "subtract_neg": np.nanmean([metric.compute_log_odds(l) for l in sub_neg_logits]),
        }

    print_scores_table(results)

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Top-Token Analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Top-Token Analysis (3 sample prompts per set)")
    print("=" * 70)

    sample_pos = val_pos[:3]
    sample_neg = val_neg[:3]
    pos_labels = [f"harmful-{i+1}" for i in range(len(sample_pos))]
    neg_labels = [f"benign-{i+1}" for i in range(len(sample_neg))]

    # Baseline
    show_top_tokens(evaluator, sample_pos, pos_labels, title="Baseline (harmful)")
    show_top_tokens(evaluator, sample_neg, neg_labels, title="Baseline (benign)")

    # Per-layer: ablation on harmful, addition on benign
    for layer, direction in directions.items():
        show_top_tokens(
            evaluator, sample_pos, pos_labels,
            intervention=(direction, "ablate", list(range(num_layers)), 1.0),
            title=f"Layer {layer} global ablation (harmful)"
        )
        show_top_tokens(
            evaluator, sample_neg, neg_labels,
            intervention=(direction, "add", [layer], 1.0),
            title=f"Layer {layer} addition (benign)"
        )

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Mini Eyeball (32 tokens)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Mini Eyeball (32 tokens, 3 prompts per condition)")
    print("=" * 70)

    eyeball_pos = eval_pos[:3]
    eyeball_neg = eval_neg[:3]

    for layer, direction in directions.items():
        print(f"\n{'─'*60}")
        print(f"Layer {layer}")
        print(f"{'─'*60}")

        print(f"\n  Harmful prompts — baseline vs global ablation:")
        gen = framework.suite.test_generation(direction, eyeball_pos, "ablate", max_new_tokens=32)
        for condition, entries in gen.items():
            print(f"    [{condition}]")
            for e in entries:
                print(f"      Q: {e['prompt'][:65]}...")
                print(f"      A: {e['generated_text'][:150]}")

        print(f"\n  Benign prompts — baseline vs layer-specific addition:")
        gen = framework.suite.test_generation(direction, eyeball_neg, "add", max_new_tokens=32)
        for condition, entries in gen.items():
            print(f"    [{condition}]")
            for e in entries:
                print(f"      Q: {e['prompt'][:65]}...")
                print(f"      A: {e['generated_text'][:150]}")

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 5: Follow-ups (if induction failed)
    # ══════════════════════════════════════════════════════════════════════
    best_induce = max(results[l]["induce"] for l in results)
    delta_from_baseline = best_induce - baseline_neg_lo
    induction_failed = delta_from_baseline < 0.5

    if not induction_failed:
        print("\n" + "=" * 70)
        print(f"INDUCTION SUCCEEDED (best induce={best_induce:.4f}, "
              f"Δ from baseline={delta_from_baseline:+.4f})")
        print("Skipping follow-up experiments.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"INDUCTION FAILED at strength=1 (best induce={best_induce:.4f}, "
              f"Δ from baseline={delta_from_baseline:+.4f})")
        print("Running follow-up experiments...")
        print("=" * 70)

        # ── 5a: Strength Sweep ───────────────────────────────────────────
        print("\n--- 5a: Strength Sweep ---")
        strengths = [1, 2, 5, 10, 20, 50]

        best_per_layer = {}  # layer -> (best_strength, best_lo)
        for layer, direction in directions.items():
            print(f"\n  Layer {layer}:")
            print(f"  {'Strength':>10s}  {'Induce LO':>12s}  {'Δ baseline':>12s}")
            print(f"  {'─'*10}  {'─'*12}  {'─'*12}")

            best_s, best_lo = 1, results[layer]["induce"]
            for s in strengths:
                logits = get_logits_custom(evaluator, val_neg, direction, "add", [layer], s)
                lo = np.nanmean([metric.compute_log_odds(l) for l in logits])
                delta = lo - baseline_neg_lo
                marker = " ← best" if lo > best_lo else ""
                print(f"  {s:>10.1f}  {lo:>12.4f}  {delta:>+12.4f}{marker}")
                if lo > best_lo:
                    best_lo = lo
                    best_s = s
            best_per_layer[layer] = (best_s, best_lo)

            # Top tokens at best strength (if it improved)
            if best_s > 1:
                show_top_tokens(
                    evaluator, sample_neg, neg_labels,
                    intervention=(direction, "add", [layer], best_s),
                    title=f"Layer {layer} addition @ strength={best_s} (benign)"
                )

        # ── 5b: Multi-Layer Injection ────────────────────────────────────
        print("\n--- 5b: Multi-Layer Injection (layers 12-15) ---")
        multi_layers = [12, 13, 14, 15]
        print(f"  {'Direction':>12s}  {'Strength':>10s}  {'Induce LO':>12s}  {'Δ baseline':>12s}")
        print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*12}")

        for layer, direction in directions.items():
            for s in [1, 5, 10]:
                logits = get_logits_custom(evaluator, val_neg, direction, "add", multi_layers, s)
                lo = np.nanmean([metric.compute_log_odds(l) for l in logits])
                delta = lo - baseline_neg_lo
                print(f"  {'Layer '+str(layer):>12s}  {s:>10.1f}  {lo:>12.4f}  {delta:>+12.4f}")

        # ── 5c: Layer 12 is already in PROBE_LAYERS ─────────────────────
        print("\n--- 5c: Layer 12 included in main comparison above ---")

        # ── 5d: Per-Prompt Induce Distribution ───────────────────────────
        print("\n--- 5d: Per-Prompt Induce Distribution ---")

        for layer, direction in directions.items():
            logits = get_logits_custom(evaluator, val_neg, direction, "add", [layer], 1.0)
            per_prompt = [metric.compute_log_odds(l) for l in logits]
            per_prompt_clean = [x for x in per_prompt if not np.isnan(x)]

            print(f"\n  Layer {layer} (n={len(per_prompt_clean)} prompts):")
            arr = np.array(per_prompt_clean)
            print(f"    Mean:   {np.mean(arr):.4f}")
            print(f"    Median: {np.median(arr):.4f}")
            print(f"    Std:    {np.std(arr):.4f}")
            print(f"    Min:    {np.min(arr):.4f}")
            print(f"    Max:    {np.max(arr):.4f}")

            # Histogram
            bins = [-16, -14, -12, -10, -8, -6, -4, -2, 0, 2]
            counts, edges = np.histogram(arr, bins=bins)
            print(f"    Distribution:")
            for j in range(len(counts)):
                bar = "█" * counts[j]
                print(f"      [{edges[j]:>5.0f}, {edges[j+1]:>5.0f}) {counts[j]:>3d} {bar}")

            # Best/worst prompts
            sorted_idx = np.argsort(per_prompt_clean)
            print(f"    Most induced (highest log-odds):")
            for idx in sorted_idx[-3:]:
                print(f"      {per_prompt_clean[idx]:>8.4f}  {val_neg[idx][:70]}")
            print(f"    Least induced (lowest log-odds):")
            for idx in sorted_idx[:3]:
                print(f"      {per_prompt_clean[idx]:>8.4f}  {val_neg[idx][:70]}")

        # ── 5e: What Addition Actually Produces at High Strength ─────────
        print("\n--- 5e: What Addition Produces at Higher Strengths ---")

        for layer, direction in directions.items():
            best_s = best_per_layer[layer][0]
            test_strengths = sorted(set([best_s, 10, 50]))
            for s in test_strengths:
                show_top_tokens(
                    evaluator, sample_neg[:2], neg_labels[:2],
                    intervention=(direction, "add", [layer], s),
                    title=f"Layer {layer} addition @ strength={s}"
                )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PROBE COMPLETE")
    print("=" * 70)
