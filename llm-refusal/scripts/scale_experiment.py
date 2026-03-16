"""
Scale experiment: refusal direction search on Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct.

Tests whether scaling to larger models improves induction, using the existing pipeline as-is.
Runs autonomously with zero user input. Expected runtime: 60-90 min on M4 Mac.

Phases:
  A: Qwen2.5-3B-Instruct (~15 min)
  B: Qwen2.5-7B-Instruct (~30-40 min)
  C: Cross-scale comparison table
  D: Deeper analysis on best model (conditional)
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc
import time
import traceback
import numpy as np
import torch as t
import torch.nn.functional as F
import logging

from framework import DirectionTestFramework
from datatypes import PromptData, DirectionVector
from scoring import Three_Score_Evaluator
from generation import generate_with_hooks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",    # ~0.5B, smallest Qwen2.5
    "google/gemma-3-1b-it",           # ~1B, Gemma architecture
    "Qwen/Qwen2.5-1.5B-Instruct",    # ~1.5B, comparable to our 1.8B baseline
    "Qwen/Qwen2.5-3B-Instruct",      # ~3B
    "Qwen/Qwen2.5-7B-Instruct",      # ~7B
    "meta-llama/Llama-3.2-3B-Instruct",  # ~3B, Llama architecture (gated)
    "meta-llama/Llama-3.1-8B-Instruct",  # ~8B, Llama architecture (gated)
]

# Hardcoded reference results from Qwen1.5-1.8B-Chat search
REFERENCE_1_8B = {
    "model": "Qwen1.5-1.8B-Chat",
    "layer": 12,
    "pos": -1,
    "bypass": -10.90,
    "induce": -11.53,
    "kl": 1.11,
    "baseline_pos_lo": -9.32,
    "baseline_neg_lo": -11.54,
    "direction_norm": 9.00,
}

# Sub-batch size for 7B model forward passes (reduces peak memory)
SUB_BATCH_SIZE_7B = 4

# Induce thresholds for deeper analysis
INDUCE_THRESHOLD_MEANINGFUL = -8.0   # Δ from baseline toward 0
INDUCE_THRESHOLD_POSITIVE = 0.0      # Actually positive induction


# ── Helpers ──────────────────────────────────────────────────────────────────

def monkey_patch_sub_batched_get_logits(evaluator, batch_size):
    """
    Monkey-patch _get_logits on a Three_Score_Evaluator to process prompts in
    sub-batches. Each sub-batch independently applies/clears interventions,
    which is correct but slightly wasteful. Acceptable for our purposes.
    """
    original_get_logits = evaluator._get_logits

    def sub_batched_get_logits(prompts, intervention=None):
        if len(prompts) <= batch_size:
            return original_get_logits(prompts, intervention)
        all_logits = []
        for i in range(0, len(prompts), batch_size):
            sub = prompts[i:i + batch_size]
            all_logits.extend(original_get_logits(sub, intervention))
        return all_logits

    evaluator._get_logits = sub_batched_get_logits
    logger.info(f"Monkey-patched _get_logits with sub-batch size {batch_size}")


def get_logits_custom(evaluator, prompts, direction, int_type, layers, strength):
    """Get logits with a custom intervention strength."""
    evaluator.intervention_applier.apply_direction_intervention(
        direction, int_type, strength=strength, layers=layers
    )
    try:
        return evaluator._get_logits(prompts)
    finally:
        evaluator.intervention_applier.clear_interventions()


def show_top_tokens(evaluator, prompts, labels, intervention=None, title="", n=10):
    """Show top-n predicted next tokens with probabilities."""
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


def free_model_memory(framework):
    """Aggressively free GPU/unified memory after finishing with a model."""
    model_name = framework.model_name
    del framework
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
    elif t.backends.mps.is_available():
        t.mps.empty_cache()
    logger.info(f"Freed memory for {model_name}")


# ── Per-Model Search + Analysis ─────────────────────────────────────────────

def run_search_for_model(model_name, use_sub_batching=False):
    """
    Full search + analysis for one model. Returns a dict with results,
    or None if the model fails to load.
    """
    model_short = model_name.split('/')[-1]
    print(f"\n{'═'*70}")
    print(f"PHASE: {model_short}")
    print(f"{'═'*70}")

    t0 = time.time()

    # ── Load model ───────────────────────────────────────────────────────
    logger.info(f"Loading {model_name}...")
    framework = DirectionTestFramework(model_name=model_name, concept="refusal")
    concept = framework.concept

    evaluator = Three_Score_Evaluator(
        framework.model, framework.tokenizer,
        framework.intervention_applier, framework.prompt_formatter,
        target_tokens=concept.target_tokens
    )
    metric = evaluator.metric
    num_layers = len(framework.intervention_applier.transformer_layers)

    # Sub-batch for large models
    if use_sub_batching:
        monkey_patch_sub_batched_get_logits(evaluator, SUB_BATCH_SIZE_7B)
        # Also patch the evaluator inside the finder so search uses sub-batching
        monkey_patch_sub_batched_get_logits(framework.finder.evaluator, SUB_BATCH_SIZE_7B)

    # ── Data (same split as 1.8B: test_size=0.2, random_state=39) ────────
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

    # ── Compute baseline log-odds ────────────────────────────────────────
    print("\nComputing baseline logits...")
    baseline_pos_logits = evaluator._get_logits(val_pos)
    baseline_neg_logits = evaluator._get_logits(val_neg)
    baseline_pos_lo = np.nanmean([metric.compute_log_odds(l) for l in baseline_pos_logits])
    baseline_neg_lo = np.nanmean([metric.compute_log_odds(l) for l in baseline_neg_logits])
    print(f"  Baseline pos log-odds: {baseline_pos_lo:.4f}")
    print(f"  Baseline neg log-odds: {baseline_neg_lo:.4f}")

    # ── Run search ───────────────────────────────────────────────────────
    print("\nRunning direction search...")
    direction = framework.finder.find_best_direction(train_data, val_data)
    if direction is None:
        logger.error("Search failed — no direction found.")
        result = {"model": model_short, "error": "no direction found"}
        free_model_memory(framework)
        return result

    # Compute final scores for the selected direction
    scores = evaluator.compute_all_scores(direction, val_data, baseline_neg_logits=baseline_neg_logits)
    direction_norm = t.norm(direction.vector).item()

    print(f"\n{'─'*60}")
    print(f"SEARCH RESULT: {model_short}")
    print(f"{'─'*60}")
    print(f"  Best layer:        {direction.layer}")
    print(f"  Best pos:          {direction.position_index}")
    print(f"  Bypass:            {scores.bypass:.4f}")
    print(f"  Induce:            {scores.induce:.4f}")
    print(f"  KL:                {scores.kl:.4f}")
    print(f"  Baseline pos LO:   {baseline_pos_lo:.4f}")
    print(f"  Baseline neg LO:   {baseline_neg_lo:.4f}")
    print(f"  Direction norm:    {direction_norm:.4f}")
    print(f"  Induce Δ baseline: {scores.induce - baseline_neg_lo:+.4f}")

    # ── Save direction ───────────────────────────────────────────────────
    save_path = f"results/{model_short}-refusal-direction"
    os.makedirs("results", exist_ok=True)
    direction.save(save_path)
    print(f"  Saved to: {save_path}.pt/.json")

    # ── Top-token analysis (3 harmful + 3 benign) ────────────────────────
    print(f"\n{'─'*60}")
    print(f"TOP-TOKEN ANALYSIS: {model_short}")
    print(f"{'─'*60}")

    sample_pos = val_pos[:3]
    sample_neg = val_neg[:3]
    pos_labels = [f"harmful-{i+1}" for i in range(len(sample_pos))]
    neg_labels = [f"benign-{i+1}" for i in range(len(sample_neg))]

    # Baseline
    show_top_tokens(evaluator, sample_pos, pos_labels, title="Baseline (harmful)")
    show_top_tokens(evaluator, sample_neg, neg_labels, title="Baseline (benign)")

    # With best direction: ablation on harmful, addition on benign
    show_top_tokens(
        evaluator, sample_pos, pos_labels,
        intervention=(direction, "ablate", list(range(num_layers)), 1.0),
        title=f"Layer {direction.layer} global ablation (harmful)"
    )
    show_top_tokens(
        evaluator, sample_neg, neg_labels,
        intervention=(direction, "add", [direction.layer], 1.0),
        title=f"Layer {direction.layer} addition (benign)"
    )

    # ── Mini eyeball if induce improved ──────────────────────────────────
    induce_delta = scores.induce - baseline_neg_lo
    if induce_delta > 0.5:
        print(f"\n{'─'*60}")
        print(f"MINI EYEBALL (induce Δ={induce_delta:+.4f}): {model_short}")
        print(f"{'─'*60}")

        # Harmful prompts — ablation
        print(f"\n  Harmful prompts — baseline vs global ablation:")
        gen = framework.suite.test_generation(direction, eval_pos[:3], "ablate", max_new_tokens=32)
        for condition, entries in gen.items():
            print(f"    [{condition}]")
            for e in entries:
                print(f"      Q: {e['prompt'][:65]}...")
                print(f"      A: {e['generated_text'][:150]}")

        # Benign prompts — addition
        print(f"\n  Benign prompts — baseline vs layer-specific addition:")
        gen = framework.suite.test_generation(direction, eval_neg[:3], "add", max_new_tokens=32)
        for condition, entries in gen.items():
            print(f"    [{condition}]")
            for e in entries:
                print(f"      Q: {e['prompt'][:65]}...")
                print(f"      A: {e['generated_text'][:150]}")

    elapsed = time.time() - t0
    print(f"\n  Phase completed in {elapsed/60:.1f} min")

    # ── Diagnostic strength sweep (always run, cheap) ──────────────────
    print(f"\n--- Diagnostic Strength Sweep ---")
    print(f"  {'Strength':>10s}  {'Induce LO':>12s}  {'Δ baseline':>12s}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}")
    for s in [1, 2, 5, 10, 20, 50]:
        logits = get_logits_custom(evaluator, val_neg, direction, "add", [direction.layer], s)
        lo = np.nanmean([metric.compute_log_odds(l) for l in logits])
        delta = lo - baseline_neg_lo
        print(f"  {s:>10.1f}  {lo:>12.4f}  {delta:>+12.4f}")

    elapsed = time.time() - t0
    print(f"\n  Phase completed in {elapsed/60:.1f} min")

    # Build result dict with only serializable data (no model references)
    result = {
        "model": model_short,
        "model_name": model_name,
        "layer": direction.layer,
        "pos": direction.position_index,
        "bypass": scores.bypass,
        "induce": scores.induce,
        "kl": scores.kl,
        "baseline_pos_lo": baseline_pos_lo,
        "baseline_neg_lo": baseline_neg_lo,
        "direction_norm": direction_norm,
        "num_layers": num_layers,
        "elapsed_min": elapsed / 60,
    }

    # Free memory eagerly — critical when running many models sequentially
    free_model_memory(framework)

    return result


# ── Cross-Scale Comparison ───────────────────────────────────────────────────

def print_cross_scale_table(all_results):
    """Print the final cross-scale comparison table."""
    print(f"\n{'═'*70}")
    print(f"CROSS-SCALE COMPARISON")
    print(f"{'═'*70}")

    models = [REFERENCE_1_8B["model"]] + [r["model"] for r in all_results]
    col_width = 18

    # Header
    print(f"  {'Metric':<20s}", end="")
    for m in models:
        print(f"  {m:>{col_width}s}", end="")
    print()
    print(f"  {'─'*20}", end="")
    for _ in models:
        print(f"  {'─'*col_width}", end="")
    print()

    rows = [
        ("Best layer", "layer"),
        ("Best pos", "pos"),
        ("Bypass", "bypass"),
        ("Induce", "induce"),
        ("KL", "kl"),
        ("Baseline pos LO", "baseline_pos_lo"),
        ("Baseline neg LO", "baseline_neg_lo"),
        ("Direction norm", "direction_norm"),
    ]

    for display, key in rows:
        print(f"  {display:<20s}", end="")
        # Reference 1.8B
        val = REFERENCE_1_8B.get(key)
        if isinstance(val, float):
            print(f"  {val:>{col_width}.4f}", end="")
        elif val is not None:
            print(f"  {val:>{col_width}}", end="")
        else:
            print(f"  {'N/A':>{col_width}s}", end="")

        # Other models
        for r in all_results:
            if "error" in r:
                print(f"  {'ERROR':>{col_width}s}", end="")
            else:
                val = r.get(key)
                if isinstance(val, float):
                    print(f"  {val:>{col_width}.4f}", end="")
                elif val is not None:
                    print(f"  {val:>{col_width}}", end="")
                else:
                    print(f"  {'N/A':>{col_width}s}", end="")
        print()

    # Induce delta row
    print(f"  {'Induce Δ baseline':<20s}", end="")
    ref_delta = REFERENCE_1_8B["induce"] - REFERENCE_1_8B["baseline_neg_lo"]
    print(f"  {ref_delta:>+{col_width}.4f}", end="")
    for r in all_results:
        if "error" in r:
            print(f"  {'ERROR':>{col_width}s}", end="")
        else:
            delta = r["induce"] - r["baseline_neg_lo"]
            print(f"  {delta:>+{col_width}.4f}", end="")
    print()


# ── Deeper Analysis (Conditional) ────────────────────────────────────────────

def deeper_analysis(result):
    """
    Run deeper analysis if induce score is meaningfully improved.
    Operates on the result dict from run_search_for_model (which still has
    the framework and evaluator loaded).
    """
    model_short = result["model"]
    induce = result["induce"]
    baseline_neg_lo = result["baseline_neg_lo"]
    direction = result["direction"]
    evaluator = result["evaluator"]
    framework = result["framework"]
    val_neg = result["val_neg"]
    val_pos = result["val_pos"]
    eval_pos = result["eval_pos"]
    eval_neg = result["eval_neg"]
    num_layers = result["num_layers"]
    metric = evaluator.metric

    print(f"\n{'═'*70}")
    print(f"DEEPER ANALYSIS: {model_short}")
    print(f"{'═'*70}")

    # ── Strength sweep ───────────────────────────────────────────────────
    print(f"\n--- Strength Sweep ---")
    strengths = [1, 2, 5, 10, 20]
    print(f"  {'Strength':>10s}  {'Induce LO':>12s}  {'Δ baseline':>12s}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}")

    best_strength, best_lo = 1, induce
    for s in strengths:
        logits = get_logits_custom(evaluator, val_neg, direction, "add", [direction.layer], s)
        lo = np.nanmean([metric.compute_log_odds(l) for l in logits])
        delta = lo - baseline_neg_lo
        marker = " ← best" if lo > best_lo else ""
        print(f"  {s:>10.1f}  {lo:>12.4f}  {delta:>+12.4f}{marker}")
        if lo > best_lo:
            best_lo = lo
            best_strength = s

    # ── Per-prompt induce distribution ────────────────────────────────────
    print(f"\n--- Per-Prompt Induce Distribution ---")
    logits = get_logits_custom(evaluator, val_neg, direction, "add", [direction.layer], 1.0)
    per_prompt = [metric.compute_log_odds(l) for l in logits]
    per_prompt_clean = [x for x in per_prompt if not np.isnan(x)]
    arr = np.array(per_prompt_clean)

    print(f"  n={len(per_prompt_clean)} prompts")
    print(f"  Mean:   {np.mean(arr):.4f}")
    print(f"  Median: {np.median(arr):.4f}")
    print(f"  Std:    {np.std(arr):.4f}")
    print(f"  Min:    {np.min(arr):.4f}")
    print(f"  Max:    {np.max(arr):.4f}")

    # Histogram
    bins = [-16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4]
    counts, edges = np.histogram(arr, bins=bins)
    print(f"  Distribution:")
    for j in range(len(counts)):
        bar = "█" * counts[j]
        print(f"    [{edges[j]:>5.0f}, {edges[j+1]:>5.0f}) {counts[j]:>3d} {bar}")

    # Best/worst prompts
    sorted_idx = np.argsort(per_prompt_clean)
    print(f"  Most induced (highest log-odds):")
    for idx in sorted_idx[-3:]:
        print(f"    {per_prompt_clean[idx]:>8.4f}  {val_neg[idx][:70]}")
    print(f"  Least induced (lowest log-odds):")
    for idx in sorted_idx[:3]:
        print(f"    {per_prompt_clean[idx]:>8.4f}  {val_neg[idx][:70]}")

    # ── Mini eyeball at best strength ────────────────────────────────────
    print(f"\n--- Mini Eyeball (3 benign prompts, addition @ strength={best_strength}) ---")
    eyeball_prompts = eval_neg[:3]
    framework.intervention_applier.apply_direction_intervention(
        direction, "add", strength=best_strength, layers=[direction.layer]
    )
    try:
        texts = generate_with_hooks(
            framework.model, framework.tokenizer,
            framework.prompt_formatter, eyeball_prompts,
            max_new_tokens=32
        )
    finally:
        framework.intervention_applier.clear_interventions()

    for prompt, text in zip(eyeball_prompts, texts):
        print(f"  Q: {prompt[:65]}...")
        print(f"  A: {text[:150]}")
        print()

    # ── Extended analysis if induce > 0 ──────────────────────────────────
    if induce > INDUCE_THRESHOLD_POSITIVE:
        print(f"\n--- Extended Eyeball (induce > 0: {induce:.4f}) ---")
        extended_prompts = eval_neg[:5]
        framework.intervention_applier.apply_direction_intervention(
            direction, "add", strength=1.0, layers=[direction.layer]
        )
        try:
            texts = generate_with_hooks(
                framework.model, framework.tokenizer,
                framework.prompt_formatter, extended_prompts,
                max_new_tokens=64
            )
        finally:
            framework.intervention_applier.clear_interventions()

        for prompt, text in zip(extended_prompts, texts):
            print(f"  Q: {prompt[:65]}...")
            print(f"  A: {text[:200]}")
            print()

        # Cross-layer induce profile
        print(f"\n--- Cross-Layer Induce Profile (pos=-1) ---")
        print(f"  {'Layer':>6s}  {'Induce LO':>12s}  {'Δ baseline':>12s}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*12}")

        for layer_idx in range(num_layers):
            # Create a temporary direction at this layer
            # Re-use the same vector (from the best layer) — this tests
            # if the SAME direction has different induction power at different layers
            temp_dir = DirectionVector(
                vector=direction.vector, layer=layer_idx,
                position_index=direction.position_index, score=0.0
            )
            logits = get_logits_custom(evaluator, val_neg, temp_dir, "add", [layer_idx], 1.0)
            lo = np.nanmean([metric.compute_log_odds(l) for l in logits])
            delta = lo - baseline_neg_lo
            marker = " ← BEST LAYER" if layer_idx == direction.layer else ""
            print(f"  {layer_idx:>6d}  {lo:>12.4f}  {delta:>+12.4f}{marker}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("SCALE EXPERIMENT: Refusal Direction Search on Larger Models")
    print(f"Models: {', '.join(m.split('/')[-1] for m in MODELS)}")
    print(f"Reference: {REFERENCE_1_8B['model']} (hardcoded from prior search)")
    print("=" * 70)

    total_t0 = time.time()
    all_results = []

    for i, model_name in enumerate(MODELS):
        model_short = model_name.split('/')[-1]
        # Sub-batch for models >= ~7B params to avoid OOM
        use_sub_batching = any(tag in model_short for tag in ["7B", "7b", "8B", "8b"])

        try:
            result = run_search_for_model(model_name, use_sub_batching=use_sub_batching)
            all_results.append(result)
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "mps" in error_msg.lower():
                logger.error(f"OOM loading/running {model_short}: {error_msg}")
                all_results.append({"model": model_short, "error": f"OOM: {error_msg[:200]}"})
            else:
                logger.error(f"Error with {model_short}: {error_msg}")
                logger.error(traceback.format_exc())
                all_results.append({"model": model_short, "error": error_msg[:200]})
        except Exception as e:
            logger.error(f"Unexpected error with {model_short}: {e}")
            logger.error(traceback.format_exc())
            all_results.append({"model": model_short, "error": str(e)[:200]})

    # ── Phase C: Cross-Scale Comparison ──────────────────────────────────
    # Print comparison table with all models (including failures)
    comparison_results = []
    for r in all_results:
        # For the table, extract just the numeric fields
        if "error" not in r:
            comparison_results.append({
                "model": r["model"],
                "layer": r["layer"],
                "pos": r["pos"],
                "bypass": r["bypass"],
                "induce": r["induce"],
                "kl": r["kl"],
                "baseline_pos_lo": r["baseline_pos_lo"],
                "baseline_neg_lo": r["baseline_neg_lo"],
                "direction_norm": r["direction_norm"],
            })
        else:
            comparison_results.append(r)

    print_cross_scale_table(comparison_results)

    # ── Summary ──────────────────────────────────────────────────────────
    best_result = None
    best_induce_delta = float('-inf')
    for r in all_results:
        if "error" in r:
            continue
        delta = r["induce"] - r["baseline_neg_lo"]
        if delta > best_induce_delta:
            best_induce_delta = delta
            best_result = r

    if best_result is not None:
        if best_result["induce"] > INDUCE_THRESHOLD_POSITIVE:
            print(f"\n  POSITIVE INDUCTION achieved by {best_result['model']}! "
                  f"(induce={best_result['induce']:.4f})")
        elif best_result["induce"] > INDUCE_THRESHOLD_MEANINGFUL:
            print(f"\n  Meaningful induction improvement by {best_result['model']} "
                  f"(induce={best_result['induce']:.4f}, Δ baseline={best_induce_delta:+.4f})")
        else:
            print(f"\n  No model achieved induce > {INDUCE_THRESHOLD_MEANINGFUL}. "
                  f"Best: {best_result['model']} (induce={best_result['induce']:.4f}, "
                  f"Δ baseline={best_induce_delta:+.4f})")
            print(f"  Pipeline fixes needed — see future_plans.md")

    total_elapsed = time.time() - total_t0
    print(f"\n{'═'*70}")
    print(f"SCALE EXPERIMENT COMPLETE")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"{'═'*70}")
