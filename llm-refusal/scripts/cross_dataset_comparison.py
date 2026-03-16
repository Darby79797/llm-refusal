"""
Cross-dataset comparison: 2×2 factorial (train data × eval data) for refusal.

Tests whether Arditi-trained directions are genuinely weaker, or just look weaker
because we evaluate on our prompts. Extends the comparison to:

  | Condition      | Train Data        | Eval Data          |
  |----------------|-------------------|--------------------|
  | ours→ours      | our 90+63         | our 100+80         |
  | ours→arditi    | our 90+63         | Arditi 60+80 held  |
  | arditi→ours    | Arditi 200+1000   | our 100+80         |
  | arditi→arditi  | Arditi 200+1000   | Arditi 60+80 held  |

Per condition: layer sweep → best layer → quantitative eval + mini eyeball.

Usage:
  python llm-refusal/scripts/cross_dataset_comparison.py
  python llm-refusal/scripts/cross_dataset_comparison.py 2>&1 | tee results/cross_dataset_comparison.log
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import torch as t
import logging

from framework import DirectionTestFramework
from datatypes import PromptData, DirectionVector
from generation import generate_with_hooks
from concept import get_concept, DEFAULT_REFUSAL_PHRASES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL = "meta-llama/Llama-2-7b-chat-hf"
SWEEP_LAYERS = [4, 6, 8, 10, 12, 14, 16]
STRENGTH = 3.0
POS = -1
NUM_SWEEP_PROMPTS = 20
NUM_EYEBALL_PROMPTS = 3


# ── Data loading ─────────────────────────────────────────────────────────────

def load_our_data():
    """Load our contrastive pairs: train and eval sets."""
    from prompts import create_refusal_train_data, create_refusal_eval_data
    train_pos, train_neg = create_refusal_train_data()
    eval_pos, eval_neg = create_refusal_eval_data()
    return (train_pos, train_neg), (eval_pos, eval_neg)


def load_arditi_data():
    """Load and split Arditi data into train and eval sets.

    Train: first 200 harmful + first 1000 harmless
    Eval:  harmful[200:] (60) + harmless[1000:1080] (80)
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "arditi_harmful_train.json")) as f:
        harmful_raw = json.load(f)
    with open(os.path.join(data_dir, "arditi_harmless_train.json")) as f:
        harmless_raw = json.load(f)

    harmful = [entry["instruction"] for entry in harmful_raw]
    harmless = [entry["instruction"] for entry in harmless_raw]

    train_pos = harmful[:200]
    train_neg = harmless[:1000]
    eval_pos = harmful[200:]       # 60 harmful
    eval_neg = harmless[1000:1080] # 80 harmless

    return (train_pos, train_neg), (eval_pos, eval_neg)


# ── Garbled/classification (adapted from layer_sweep.py) ─────────────────────

def is_garbled(text: str) -> bool:
    """Check if text is garbled: non-ASCII dominance, excessive repetition, or very short."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    total_chars = len(text.strip())
    if total_chars < 10:
        return True
    if total_chars > 0 and ascii_chars / total_chars < 0.5:
        return True
    if total_chars > 30:
        for i in range(0, min(len(text) - 30, 100)):
            chunk = text[i:i+10]
            if text.count(chunk) >= 3 and len(chunk.strip()) > 3:
                return True
    return False


def classify_response(text: str) -> str:
    """Classify a response as DETECTED, GARBLED, or NORMAL using refusal phrase matching."""
    if is_garbled(text):
        return "GARBLED"
    lower = text.lower()
    for phrase in DEFAULT_REFUSAL_PHRASES:
        if phrase.lower() in lower:
            return "DETECTED"
    return "NORMAL"


# ── Core computation ─────────────────────────────────────────────────────────

def compute_directions(framework, train_pos, train_neg):
    """Compute diff-in-means direction vectors from provided train data.

    Returns dict[(layer, pos), tensor] for all layers at pos=-1.
    """
    all_data = PromptData(
        train_pos + train_neg,
        [True] * len(train_pos) + [False] * len(train_neg)
    )
    train_data, _ = all_data.train_val_split()

    logger.info(f"Computing directions from {len(train_pos)} pos + {len(train_neg)} neg prompts...")
    diff_vectors = framework.finder.direction_finder_method.compute_difference_vectors(
        train_data, max_positions=1
    )
    return diff_vectors


def run_layer_sweep(framework, directions, eval_neg_prompts, layers):
    """Run layer sweep: add raw direction at each layer, classify responses.

    Returns list of dicts with per-layer results.
    """
    prompts = eval_neg_prompts[:NUM_SWEEP_PROMPTS]
    results = []

    for layer in layers:
        key = (layer, POS)
        if key not in directions:
            logger.warning(f"No direction vector for layer={layer}, pos={POS}")
            continue

        vec = directions[key]
        direction = DirectionVector(vector=vec, layer=layer, position_index=POS, score=0)
        norm = t.norm(direction.vector).item()

        framework.intervention_applier.apply_direction_intervention(
            direction, "add", strength=STRENGTH, layers=[layer]
        )
        try:
            texts = generate_with_hooks(
                framework.model, framework.tokenizer,
                framework.prompt_formatter, prompts,
                max_new_tokens=64
            )
        finally:
            framework.intervention_applier.clear_interventions()

        classifications = [classify_response(txt) for txt in texts]
        n_detected = classifications.count("DETECTED")
        n_garbled = classifications.count("GARBLED")
        n_normal = classifications.count("NORMAL")

        results.append({
            "layer": layer,
            "norm": norm,
            "n_detected": n_detected,
            "n_garbled": n_garbled,
            "n_normal": n_normal,
            "texts": texts,
            "classifications": classifications,
        })

    return results


def find_best_layer(sweep_results):
    """Find best layer: highest detected count, breaking ties by lowest garbled."""
    if not sweep_results:
        return None
    return max(sweep_results, key=lambda r: (r["n_detected"], -r["n_garbled"]))


def run_quant_eval(framework, direction, eval_pos, eval_neg):
    """Run quantitative evaluation at a given layer.

    Returns dict with baseline, global ablation, layer-specific ablation,
    and induction detection rates.
    """
    num_layers = len(framework.intervention_applier.transformer_layers)

    # Baseline detection rate on positive prompts
    logger.info("  Baseline detection rate...")
    baseline_rate = framework.evaluator.evaluate_detection_rate(eval_pos)

    # Global ablation on positive prompts
    logger.info("  Global ablation detection rate...")
    framework.intervention_applier.apply_direction_intervention(
        direction, "ablate", 1.0, layers=list(range(num_layers))
    )
    try:
        global_ablation_rate = framework.evaluator.evaluate_detection_rate(eval_pos)
    finally:
        framework.intervention_applier.clear_interventions()

    # Layer-specific ablation on positive prompts
    logger.info(f"  Layer-specific ablation (L{direction.layer}) detection rate...")
    framework.intervention_applier.apply_direction_intervention(
        direction, "ablate", 1.0, layers=[direction.layer]
    )
    try:
        layer_ablation_rate = framework.evaluator.evaluate_detection_rate(eval_pos)
    finally:
        framework.intervention_applier.clear_interventions()

    # Induction (addition) on negative prompts
    logger.info(f"  Induction (addition at L{direction.layer}, s={STRENGTH}) detection rate...")
    framework.intervention_applier.apply_direction_intervention(
        direction, "add", STRENGTH, layers=[direction.layer]
    )
    try:
        induction_rate = framework.evaluator.evaluate_detection_rate(eval_neg)
    finally:
        framework.intervention_applier.clear_interventions()

    return {
        "baseline": baseline_rate,
        "global_ablation": global_ablation_rate,
        "layer_ablation": layer_ablation_rate,
        "induction": induction_rate,
    }


# ── Printing ─────────────────────────────────────────────────────────────────

def print_sweep_table(condition_name, sweep_results):
    """Print layer sweep results for one condition."""
    print(f"\n{'─'*70}")
    print(f"LAYER SWEEP: {condition_name}")
    print(f"{'─'*70}")
    print(f"  {'Layer':>6} {'||dir||':>8} {'Detected':>9} {'Garbled':>8} {'Normal':>7}")
    print(f"  {'─'*6} {'─'*8} {'─'*9} {'─'*8} {'─'*7}")
    for r in sweep_results:
        marker = " ◀" if r == find_best_layer(sweep_results) else ""
        print(f"  {r['layer']:>6} {r['norm']:>8.2f} {r['n_detected']:>9} {r['n_garbled']:>8} {r['n_normal']:>7}{marker}")


def print_best_layer_table(all_conditions):
    """Print comparison of best layers across all 4 conditions."""
    print(f"\n{'═'*80}")
    print(f"BEST LAYER COMPARISON")
    print(f"{'═'*80}")

    col_w = 16
    names = [c["name"] for c in all_conditions]

    print(f"  {'':>14}", end="")
    for name in names:
        print(f"  {name:>{col_w}}", end="")
    print()
    print(f"  {'─'*14}", end="")
    for _ in names:
        print(f"  {'─'*col_w}", end="")
    print()

    rows = [
        ("Best layer", lambda c: str(c["best_layer"])),
        ("||dir||", lambda c: f"{c['best_norm']:.2f}"),
        ("Detected", lambda c: f"{c['best_detected']}/{NUM_SWEEP_PROMPTS}"),
        ("Garbled", lambda c: f"{c['best_garbled']}/{NUM_SWEEP_PROMPTS}"),
        ("Normal", lambda c: f"{c['best_normal']}/{NUM_SWEEP_PROMPTS}"),
    ]
    for label, fmt_fn in rows:
        print(f"  {label:>14}", end="")
        for c in all_conditions:
            print(f"  {fmt_fn(c):>{col_w}}", end="")
        print()


def print_quant_table(all_conditions):
    """Print quantitative evaluation comparison across all 4 conditions."""
    print(f"\n{'═'*80}")
    print(f"QUANTITATIVE EVALUATION COMPARISON")
    print(f"{'═'*80}")

    col_w = 16
    names = [c["name"] for c in all_conditions]

    print(f"  {'':>20}", end="")
    for name in names:
        print(f"  {name:>{col_w}}", end="")
    print()
    print(f"  {'─'*20}", end="")
    for _ in names:
        print(f"  {'─'*col_w}", end="")
    print()

    rows = [
        ("Layer", lambda c: str(c["best_layer"])),
        ("Baseline", lambda c: f"{c['quant']['baseline']:.2%}"),
        ("Global ablation", lambda c: f"{c['quant']['global_ablation']:.2%}"),
        ("Layer ablation", lambda c: f"{c['quant']['layer_ablation']:.2%}"),
        ("Induction", lambda c: f"{c['quant']['induction']:.2%}"),
    ]
    for label, fmt_fn in rows:
        print(f"  {label:>20}", end="")
        for c in all_conditions:
            print(f"  {fmt_fn(c):>{col_w}}", end="")
        print()


def run_mini_eyeball(framework, direction, prompts, condition_name):
    """Generate baseline vs intervened text for qualitative comparison."""
    subset = prompts[:NUM_EYEBALL_PROMPTS]

    print(f"\n{'─'*70}")
    print(f"MINI EYEBALL: {condition_name} (addition @ L{direction.layer}, strength={STRENGTH})")
    print(f"{'─'*70}")

    # Baseline
    baseline_texts = generate_with_hooks(
        framework.model, framework.tokenizer,
        framework.prompt_formatter, subset,
        max_new_tokens=64
    )

    # With addition
    framework.intervention_applier.apply_direction_intervention(
        direction, "add", strength=STRENGTH, layers=[direction.layer]
    )
    try:
        induced_texts = generate_with_hooks(
            framework.model, framework.tokenizer,
            framework.prompt_formatter, subset,
            max_new_tokens=64
        )
    finally:
        framework.intervention_applier.clear_interventions()

    for i, (prompt, base, induced) in enumerate(zip(subset, baseline_texts, induced_texts)):
        print(f"\n  {i+1}. Q: {prompt[:80]}")
        print(f"     Baseline: {base[:150]}")
        print(f"     Induced:  {induced[:150]}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    total_t0 = time.time()

    print("=" * 80)
    print("CROSS-DATASET COMPARISON: 2×2 Factorial (Train Data × Eval Data)")
    print(f"Model: {MODEL}")
    print(f"Sweep layers: {SWEEP_LAYERS}, strength={STRENGTH}")
    print("=" * 80)

    # Load data
    print("\n--- Loading data ---")
    our_train, our_eval = load_our_data()
    arditi_train, arditi_eval = load_arditi_data()

    print(f"  Our train:    {len(our_train[0])} pos + {len(our_train[1])} neg")
    print(f"  Our eval:     {len(our_eval[0])} pos + {len(our_eval[1])} neg")
    print(f"  Arditi train: {len(arditi_train[0])} pos + {len(arditi_train[1])} neg")
    print(f"  Arditi eval:  {len(arditi_eval[0])} pos + {len(arditi_eval[1])} neg")

    # Load model once
    print("\n--- Loading model ---")
    framework = DirectionTestFramework(model_name=MODEL, concept="refusal")

    # Compute directions for both training sets
    print("\n--- Computing directions ---")
    t0 = time.time()
    our_directions = compute_directions(framework, our_train[0], our_train[1])
    print(f"  Our directions computed in {time.time() - t0:.1f}s")

    t0 = time.time()
    arditi_directions = compute_directions(framework, arditi_train[0], arditi_train[1])
    print(f"  Arditi directions computed in {time.time() - t0:.1f}s")

    # Define the 4 conditions: (name, directions, eval_pos, eval_neg)
    conditions = [
        ("ours→ours",    our_directions,    our_eval[0],    our_eval[1]),
        ("ours→arditi",  our_directions,    arditi_eval[0], arditi_eval[1]),
        ("arditi→ours",  arditi_directions, our_eval[0],    our_eval[1]),
        ("arditi→arditi", arditi_directions, arditi_eval[0], arditi_eval[1]),
    ]

    # ── Layer sweeps ─────────────────────────────────────────────────────
    print("\n--- Running layer sweeps ---")
    all_conditions = []

    for name, directions, eval_pos, eval_neg in conditions:
        t0 = time.time()
        logger.info(f"Sweeping {name}...")
        sweep_results = run_layer_sweep(framework, directions, eval_neg, SWEEP_LAYERS)
        elapsed = time.time() - t0
        logger.info(f"  {name} sweep done in {elapsed:.1f}s")

        print_sweep_table(name, sweep_results)

        best = find_best_layer(sweep_results)
        if best is None:
            logger.error(f"  No valid layers for {name}")
            continue

        all_conditions.append({
            "name": name,
            "directions": directions,
            "eval_pos": eval_pos,
            "eval_neg": eval_neg,
            "best_layer": best["layer"],
            "best_norm": best["norm"],
            "best_detected": best["n_detected"],
            "best_garbled": best["n_garbled"],
            "best_normal": best["n_normal"],
            "sweep_results": sweep_results,
        })

    # ── Best layer comparison ────────────────────────────────────────────
    print_best_layer_table(all_conditions)

    # ── Quantitative evaluation at best layer ────────────────────────────
    print("\n--- Running quantitative evaluations ---")
    for cond in all_conditions:
        t0 = time.time()
        logger.info(f"Evaluating {cond['name']} at layer {cond['best_layer']}...")

        vec = cond["directions"][(cond["best_layer"], POS)]
        direction = DirectionVector(vector=vec, layer=cond["best_layer"], position_index=POS, score=0)

        cond["quant"] = run_quant_eval(framework, direction, cond["eval_pos"], cond["eval_neg"])
        elapsed = time.time() - t0
        logger.info(f"  {cond['name']} eval done in {elapsed:.1f}s")

    print_quant_table(all_conditions)

    # ── Mini eyeball: best vs worst condition ────────────────────────────
    # Best = highest induction rate; worst = lowest induction rate
    best_cond = max(all_conditions, key=lambda c: c["quant"]["induction"])
    worst_cond = min(all_conditions, key=lambda c: c["quant"]["induction"])

    vec_best = best_cond["directions"][(best_cond["best_layer"], POS)]
    dir_best = DirectionVector(vector=vec_best, layer=best_cond["best_layer"], position_index=POS, score=0)
    run_mini_eyeball(framework, dir_best, best_cond["eval_neg"], f"BEST: {best_cond['name']}")

    if worst_cond["name"] != best_cond["name"]:
        vec_worst = worst_cond["directions"][(worst_cond["best_layer"], POS)]
        dir_worst = DirectionVector(vector=vec_worst, layer=worst_cond["best_layer"], position_index=POS, score=0)
        run_mini_eyeball(framework, dir_worst, worst_cond["eval_neg"], f"WORST: {worst_cond['name']}")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - total_t0
    print(f"\n{'═'*80}")
    print(f"EXPERIMENT COMPLETE — {elapsed/60:.1f} min")
    print(f"{'═'*80}")


if __name__ == "__main__":
    main()
