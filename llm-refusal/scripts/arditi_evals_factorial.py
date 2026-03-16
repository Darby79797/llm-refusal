"""
Arditi-style evals on the 2×2 factorial (train data × eval data) for refusal.

Reuses known best layers from prior sweep (results_summary.md), skipping the
layer sweep. Computes DifferenceInMeans directions, then runs:
  - Phrase-match detection rate (existing)
  - LlamaGuard2 classification (if API configured)
  - JailbreakBench classification (if installed + API key)
  - Alpaca CE loss / perplexity (offline, always runs)

Factorial design:
  | Condition      | Train Data        | Eval Data          | Best Layer |
  |----------------|-------------------|--------------------|------------|
  | ours→ours      | our 90+63         | our 100+80         | 18         |
  | ours→arditi    | our 90+63         | Arditi 60+80 held  | 18         |
  | arditi→ours    | Arditi 200+1000   | our 100+80         | 22         |
  | arditi→arditi  | Arditi 200+1000   | Arditi 60+80 held  | 16         |

Usage:
  python llm-refusal/scripts/arditi_evals_factorial.py
  python llm-refusal/scripts/arditi_evals_factorial.py 2>&1 | tee results/arditi_evals_factorial.log
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
from evaluation import BigEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen2.5-3B-Instruct"
POS = -1
STRENGTH = 1.0
ALPACA_MAX_PROMPTS = 500

# Known best layers from prior sweep (results_summary.md)
BEST_LAYERS = {
    "ours→ours": 18,
    "ours→arditi": 18,
    "arditi→ours": 22,
    "arditi→arditi": 16,
}

# API config from environment (optional)
LLAMAGUARD_API_BASE = os.environ.get("LLAMAGUARD_API_BASE")
LLAMAGUARD_API_KEY = os.environ.get("LLAMAGUARD_API_KEY")
LLAMAGUARD_MODEL = os.environ.get("LLAMAGUARD_MODEL")
JBB_API_KEY = os.environ.get("JBB_API_KEY")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_our_data():
    """Load our contrastive pairs: train and eval sets."""
    from prompts import create_refusal_train_data, create_refusal_eval_data
    train_pos, train_neg = create_refusal_train_data()
    eval_pos, eval_neg = create_refusal_eval_data()
    return (train_pos, train_neg), (eval_pos, eval_neg)


def load_arditi_data():
    """Load and split Arditi data into train and eval sets."""
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


# ── Core computation ─────────────────────────────────────────────────────────

def compute_directions(framework, train_pos, train_neg):
    """Compute diff-in-means direction vectors from provided train data."""
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


def run_arditi_evals(framework, direction, eval_pos, eval_neg, condition_name):
    """Run all evaluation metrics for a single condition.

    Returns dict with detection rates, LlamaGuard, JBB, and Alpaca CE loss.
    """
    evaluator = BigEvaluator(
        framework,
        detection_phrases=framework.concept.detection_phrases,
        detection_fn=framework.concept.detection_fn,
        judge_prompt=framework.concept.judge_prompt,
        llamaguard_api_base=LLAMAGUARD_API_BASE,
        llamaguard_api_key=LLAMAGUARD_API_KEY,
        llamaguard_model=LLAMAGUARD_MODEL,
        jbb_api_key=JBB_API_KEY,
    )
    num_layers = len(framework.intervention_applier.transformer_layers)
    results = {}

    # ── Baseline ──
    logger.info(f"[{condition_name}] Baseline evaluation...")
    baseline_texts_pos = evaluator.generate_responses(eval_pos)
    results["baseline"] = {
        "detection_rate_pos": evaluator.evaluate_detection_rate(eval_pos, generated_texts=baseline_texts_pos),
    }
    lg = evaluator.evaluate_llamaguard_rate(eval_pos, baseline_texts_pos)
    if lg is not None:
        results["baseline"]["llamaguard_unsafe_rate_pos"] = lg
    jbb = evaluator.evaluate_jailbreakbench_rate(eval_pos, baseline_texts_pos)
    if jbb is not None:
        results["baseline"]["jbb_jailbreak_rate_pos"] = jbb
    alpaca = evaluator.evaluate_alpaca_ce_loss(max_prompts=ALPACA_MAX_PROMPTS)
    if alpaca is not None:
        results["baseline"].update(alpaca)

    # ── Global ablation ──
    logger.info(f"[{condition_name}] Global ablation evaluation...")
    framework.intervention_applier.apply_direction_intervention(
        direction, "ablate", 1.0, layers=list(range(num_layers))
    )
    try:
        ablated_texts_pos = evaluator.generate_responses(eval_pos)
        results["global_ablation"] = {
            "detection_rate_pos": evaluator.evaluate_detection_rate(eval_pos, generated_texts=ablated_texts_pos),
        }
        lg = evaluator.evaluate_llamaguard_rate(eval_pos, ablated_texts_pos)
        if lg is not None:
            results["global_ablation"]["llamaguard_unsafe_rate_pos"] = lg
        jbb = evaluator.evaluate_jailbreakbench_rate(eval_pos, ablated_texts_pos)
        if jbb is not None:
            results["global_ablation"]["jbb_jailbreak_rate_pos"] = jbb
        alpaca = evaluator.evaluate_alpaca_ce_loss(max_prompts=ALPACA_MAX_PROMPTS)
        if alpaca is not None:
            results["global_ablation"].update(alpaca)
    finally:
        framework.intervention_applier.clear_interventions()

    # ── Layer-specific ablation ──
    logger.info(f"[{condition_name}] Layer-specific ablation (L{direction.layer})...")
    framework.intervention_applier.apply_direction_intervention(
        direction, "ablate", 1.0, layers=[direction.layer]
    )
    try:
        layer_abl_texts = evaluator.generate_responses(eval_pos)
        results["layer_ablation"] = {
            "detection_rate_pos": evaluator.evaluate_detection_rate(eval_pos, generated_texts=layer_abl_texts),
        }
        lg = evaluator.evaluate_llamaguard_rate(eval_pos, layer_abl_texts)
        if lg is not None:
            results["layer_ablation"]["llamaguard_unsafe_rate_pos"] = lg
        jbb = evaluator.evaluate_jailbreakbench_rate(eval_pos, layer_abl_texts)
        if jbb is not None:
            results["layer_ablation"]["jbb_jailbreak_rate_pos"] = jbb
        alpaca = evaluator.evaluate_alpaca_ce_loss(max_prompts=ALPACA_MAX_PROMPTS)
        if alpaca is not None:
            results["layer_ablation"].update(alpaca)
    finally:
        framework.intervention_applier.clear_interventions()

    # ── Induction (addition on negative prompts) ──
    logger.info(f"[{condition_name}] Induction (addition at L{direction.layer}, s={STRENGTH})...")
    framework.intervention_applier.apply_direction_intervention(
        direction, "add", STRENGTH, layers=[direction.layer]
    )
    try:
        induced_texts = evaluator.generate_responses(eval_neg)
        results["induction"] = {
            "detection_rate_neg": evaluator.evaluate_detection_rate(eval_neg, generated_texts=induced_texts),
        }
        lg = evaluator.evaluate_llamaguard_rate(eval_neg, induced_texts)
        if lg is not None:
            results["induction"]["llamaguard_unsafe_rate_neg"] = lg
        jbb = evaluator.evaluate_jailbreakbench_rate(eval_neg, induced_texts)
        if jbb is not None:
            results["induction"]["jbb_jailbreak_rate_neg"] = jbb
    finally:
        framework.intervention_applier.clear_interventions()

    return results


# ── Printing ─────────────────────────────────────────────────────────────────

def print_results_table(all_conditions):
    """Print formatted comparison table across all 4 conditions."""
    print(f"\n{'═'*100}")
    print(f"ARDITI-STYLE EVALUATION RESULTS: {MODEL}")
    print(f"{'═'*100}")

    col_w = 18
    names = [c["name"] for c in all_conditions]

    # Header
    print(f"  {'Metric':<30s}", end="")
    for name in names:
        print(f"  {name:>{col_w}s}", end="")
    print()
    print(f"  {'─'*30}", end="")
    for _ in names:
        print(f"  {'─'*col_w}", end="")
    print()

    # Layer row
    print(f"  {'Layer':<30s}", end="")
    for c in all_conditions:
        print(f"  {c['layer']:>{col_w}}", end="")
    print()

    # Results rows, grouped by condition
    sections = [
        ("BASELINE", "baseline"),
        ("GLOBAL ABLATION", "global_ablation"),
        ("LAYER ABLATION", "layer_ablation"),
        ("INDUCTION", "induction"),
    ]
    for section_label, section_key in sections:
        print(f"\n  --- {section_label} ---")
        # Collect all metric keys across conditions for this section
        all_keys = []
        for c in all_conditions:
            for k in c["results"].get(section_key, {}):
                if k not in all_keys:
                    all_keys.append(k)

        for metric_key in all_keys:
            print(f"  {metric_key:<30s}", end="")
            for c in all_conditions:
                val = c["results"].get(section_key, {}).get(metric_key)
                if val is None:
                    print(f"  {'N/A':>{col_w}s}", end="")
                elif isinstance(val, float):
                    print(f"  {val:>{col_w}.4f}", end="")
                else:
                    print(f"  {str(val):>{col_w}s}", end="")
            print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    total_t0 = time.time()

    print("=" * 100)
    print("ARDITI-STYLE EVALS: 2×2 Factorial (Train Data × Eval Data)")
    print(f"Model: {MODEL}")
    print(f"Known best layers: {BEST_LAYERS}")
    print(f"LlamaGuard API: {'configured' if LLAMAGUARD_API_BASE else 'not configured (will skip)'}")
    print(f"JailbreakBench: {'API key set' if JBB_API_KEY else 'not configured (will skip)'}")
    print(f"Alpaca CE loss: {ALPACA_MAX_PROMPTS} prompts")
    print("=" * 100)

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

    # Print direction norms at best layers
    print("\n--- Direction norms at best layers ---")
    for name, layer in BEST_LAYERS.items():
        directions = our_directions if name.startswith("ours") else arditi_directions
        key = (layer, POS)
        if key in directions:
            norm = t.norm(directions[key]).item()
            print(f"  {name}: L{layer}, ||dir|| = {norm:.4f}")
        else:
            print(f"  {name}: L{layer} — NOT FOUND in direction vectors!")

    # Define conditions
    conditions = [
        ("ours→ours",     our_directions,    our_eval[0],    our_eval[1],    BEST_LAYERS["ours→ours"]),
        ("ours→arditi",   our_directions,    arditi_eval[0], arditi_eval[1], BEST_LAYERS["ours→arditi"]),
        ("arditi→ours",   arditi_directions, our_eval[0],    our_eval[1],    BEST_LAYERS["arditi→ours"]),
        ("arditi→arditi", arditi_directions, arditi_eval[0], arditi_eval[1], BEST_LAYERS["arditi→arditi"]),
    ]

    # ── Run evaluations ──────────────────────────────────────────────────
    print("\n--- Running Arditi-style evaluations ---")
    all_conditions = []

    for name, directions, eval_pos, eval_neg, layer in conditions:
        key = (layer, POS)
        if key not in directions:
            logger.error(f"Direction not found at layer={layer}, pos={POS} for {name}. Skipping.")
            continue

        vec = directions[key]
        direction = DirectionVector(vector=vec, layer=layer, position_index=POS, score=0)
        norm = t.norm(direction.vector).item()

        print(f"\n{'─'*80}")
        print(f"Evaluating: {name} (L{layer}, ||dir||={norm:.2f})")
        print(f"{'─'*80}")

        t0 = time.time()
        results = run_arditi_evals(framework, direction, eval_pos, eval_neg, name)
        elapsed = time.time() - t0
        print(f"  {name} done in {elapsed/60:.1f} min")

        all_conditions.append({
            "name": name,
            "layer": layer,
            "norm": norm,
            "results": results,
        })

    # ── Print comparison table ────────────────────────────────────────────
    print_results_table(all_conditions)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - total_t0
    print(f"\n{'═'*100}")
    print(f"EXPERIMENT COMPLETE — {elapsed/60:.1f} min")
    print(f"{'═'*100}")


if __name__ == "__main__":
    main()
