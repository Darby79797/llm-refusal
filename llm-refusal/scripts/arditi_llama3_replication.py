"""
Direct Arditi replication on Llama-3-8B-Instruct.

Tests: Does Arditi's data (AdvBench + Alpaca) produce a working refusal direction
on the model Arditi actually used?

Subsamples Arditi neg from 1000→200 for speed (diff-in-means converges well at 200).
Runs arditi→arditi and ours→arditi conditions for comparison.

Usage:
  python llm-refusal/scripts/arditi_llama3_replication.py 2>&1 | tee results/arditi_llama3_replication.log
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

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

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SWEEP_LAYERS = [6, 8, 10, 12, 14, 16, 18, 20]
STRENGTH = 1.0
POS = -1
NUM_SWEEP_PROMPTS = 20
NUM_EYEBALL_PROMPTS = 5
ARDITI_NEG_SUBSAMPLE = 200  # down from 1000


def load_our_data():
    from prompts import create_refusal_train_data, create_refusal_eval_data
    train_pos, train_neg = create_refusal_train_data()
    eval_pos, eval_neg = create_refusal_eval_data()
    return (train_pos, train_neg), (eval_pos, eval_neg)


def load_arditi_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "arditi_harmful_train.json")) as f:
        harmful_raw = json.load(f)
    with open(os.path.join(data_dir, "arditi_harmless_train.json")) as f:
        harmless_raw = json.load(f)

    harmful = [entry["instruction"] for entry in harmful_raw]
    harmless = [entry["instruction"] for entry in harmless_raw]

    train_pos = harmful[:200]
    train_neg = harmless[:ARDITI_NEG_SUBSAMPLE]
    eval_pos = harmful[200:]        # 60 harmful
    eval_neg = harmless[1000:1080]   # 80 harmless (disjoint from train)

    return (train_pos, train_neg), (eval_pos, eval_neg)


def is_garbled(text):
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


def classify_response(text):
    if is_garbled(text):
        return "GARBLED"
    lower = text.lower()
    for phrase in DEFAULT_REFUSAL_PHRASES:
        if phrase.lower() in lower:
            return "DETECTED"
    return "NORMAL"


def compute_directions(framework, train_pos, train_neg):
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


def run_layer_sweep(framework, directions, eval_neg_prompts, layers, strength):
    prompts = eval_neg_prompts[:NUM_SWEEP_PROMPTS]
    results = []

    for layer in layers:
        key = (layer, POS)
        if key not in directions:
            continue

        vec = directions[key]
        direction = DirectionVector(vector=vec, layer=layer, position_index=POS, score=0)
        norm = t.norm(direction.vector).item()

        framework.intervention_applier.apply_direction_intervention(
            direction, "add", strength=strength, layers=[layer]
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


def run_quant_eval(framework, direction, eval_pos, eval_neg, strength):
    num_layers = len(framework.intervention_applier.transformer_layers)

    logger.info("  Baseline detection rate...")
    baseline_rate = framework.evaluator.evaluate_detection_rate(eval_pos)

    logger.info("  Global ablation detection rate...")
    framework.intervention_applier.apply_direction_intervention(
        direction, "ablate", 1.0, layers=list(range(num_layers))
    )
    try:
        global_ablation_rate = framework.evaluator.evaluate_detection_rate(eval_pos)
    finally:
        framework.intervention_applier.clear_interventions()

    logger.info(f"  Layer ablation (L{direction.layer}) detection rate...")
    framework.intervention_applier.apply_direction_intervention(
        direction, "ablate", 1.0, layers=[direction.layer]
    )
    try:
        layer_ablation_rate = framework.evaluator.evaluate_detection_rate(eval_pos)
    finally:
        framework.intervention_applier.clear_interventions()

    logger.info(f"  Induction (addition at L{direction.layer}, s={strength}) detection rate...")
    framework.intervention_applier.apply_direction_intervention(
        direction, "add", strength, layers=[direction.layer]
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


def main():
    total_t0 = time.time()

    print("=" * 80)
    print("ARDITI REPLICATION ON LLAMA-3-8B-INSTRUCT")
    print(f"Model: {MODEL}")
    print(f"Sweep layers: {SWEEP_LAYERS}, strength={STRENGTH}")
    print(f"Arditi neg subsample: {ARDITI_NEG_SUBSAMPLE}")
    print("=" * 80)

    # Load data
    print("\n--- Loading data ---")
    our_train, our_eval = load_our_data()
    arditi_train, arditi_eval = load_arditi_data()

    print(f"  Our train:    {len(our_train[0])} pos + {len(our_train[1])} neg")
    print(f"  Our eval:     {len(our_eval[0])} pos + {len(our_eval[1])} neg")
    print(f"  Arditi train: {len(arditi_train[0])} pos + {len(arditi_train[1])} neg")
    print(f"  Arditi eval:  {len(arditi_eval[0])} pos + {len(arditi_eval[1])} neg")

    # Load model
    print("\n--- Loading model ---")
    framework = DirectionTestFramework(model_name=MODEL, concept="refusal")

    # Compute directions
    print("\n--- Computing directions ---")
    t0 = time.time()
    our_directions = compute_directions(framework, our_train[0], our_train[1])
    print(f"  Our directions computed in {time.time() - t0:.1f}s")

    t0 = time.time()
    arditi_directions = compute_directions(framework, arditi_train[0], arditi_train[1])
    print(f"  Arditi directions computed in {time.time() - t0:.1f}s")

    # Cosine similarity at each layer
    print("\n--- Direction cosine similarity per layer ---")
    for layer in SWEEP_LAYERS:
        key = (layer, POS)
        if key in our_directions and key in arditi_directions:
            cos = t.nn.functional.cosine_similarity(
                our_directions[key].unsqueeze(0),
                arditi_directions[key].unsqueeze(0)
            ).item()
            print(f"  L{layer}: cos={cos:.4f}  ||ours||={t.norm(our_directions[key]):.2f}  ||arditi||={t.norm(arditi_directions[key]):.2f}")

    # Layer sweeps
    conditions = [
        ("arditi→arditi", arditi_directions, arditi_eval[1]),
        ("ours→arditi",   our_directions,    arditi_eval[1]),
        ("ours→ours",     our_directions,    our_eval[1]),
    ]

    print("\n--- Layer sweeps (strength=1) ---")
    all_results = {}
    for name, directions, eval_neg in conditions:
        t0 = time.time()
        logger.info(f"Sweeping {name}...")
        sweep = run_layer_sweep(framework, directions, eval_neg, SWEEP_LAYERS, strength=STRENGTH)
        elapsed = time.time() - t0

        print(f"\n{'─'*70}")
        print(f"LAYER SWEEP: {name}  (strength={STRENGTH})")
        print(f"{'─'*70}")
        print(f"  {'Layer':>6} {'||dir||':>8} {'Detected':>9} {'Garbled':>8} {'Normal':>7}")
        print(f"  {'─'*6} {'─'*8} {'─'*9} {'─'*8} {'─'*7}")
        best = max(sweep, key=lambda r: (r["n_detected"], -r["n_garbled"])) if sweep else None
        for r in sweep:
            marker = " ◀" if r is best else ""
            print(f"  {r['layer']:>6} {r['norm']:>8.2f} {r['n_detected']:>9} {r['n_garbled']:>8} {r['n_normal']:>7}{marker}")

        all_results[name] = {"sweep": sweep, "best": best}
        logger.info(f"  {name} done in {elapsed:.1f}s")

    # Also try strength=3 for arditi→arditi (Arditi may have used higher strength)
    print("\n--- Layer sweep: arditi→arditi at strength=3 ---")
    sweep_s3 = run_layer_sweep(framework, arditi_directions, arditi_eval[1], SWEEP_LAYERS, strength=3.0)
    print(f"\n{'─'*70}")
    print(f"LAYER SWEEP: arditi→arditi  (strength=3)")
    print(f"{'─'*70}")
    print(f"  {'Layer':>6} {'||dir||':>8} {'Detected':>9} {'Garbled':>8} {'Normal':>7}")
    print(f"  {'─'*6} {'─'*8} {'─'*9} {'─'*8} {'─'*7}")
    best_s3 = max(sweep_s3, key=lambda r: (r["n_detected"], -r["n_garbled"])) if sweep_s3 else None
    for r in sweep_s3:
        marker = " ◀" if r is best_s3 else ""
        print(f"  {r['layer']:>6} {r['norm']:>8.2f} {r['n_detected']:>9} {r['n_garbled']:>8} {r['n_normal']:>7}{marker}")

    # Quantitative eval for arditi→arditi at best layer
    best_arditi = all_results["arditi→arditi"]["best"]
    best_ours = all_results["ours→arditi"]["best"]

    if best_arditi:
        print(f"\n--- Quantitative eval: arditi→arditi at L{best_arditi['layer']} ---")
        vec = arditi_directions[(best_arditi["layer"], POS)]
        direction = DirectionVector(vector=vec, layer=best_arditi["layer"], position_index=POS, score=0)
        arditi_quant = run_quant_eval(framework, direction, arditi_eval[0], arditi_eval[1], STRENGTH)

    if best_ours:
        print(f"\n--- Quantitative eval: ours→arditi at L{best_ours['layer']} ---")
        vec = our_directions[(best_ours["layer"], POS)]
        direction = DirectionVector(vector=vec, layer=best_ours["layer"], position_index=POS, score=0)
        ours_quant = run_quant_eval(framework, direction, arditi_eval[0], arditi_eval[1], STRENGTH)

    # Print comparison
    print(f"\n{'═'*80}")
    print(f"QUANTITATIVE COMPARISON")
    print(f"{'═'*80}")
    col_w = 18
    print(f"  {'':>20}  {'arditi→arditi':>{col_w}}  {'ours→arditi':>{col_w}}")
    print(f"  {'─'*20}  {'─'*col_w}  {'─'*col_w}")
    if best_arditi and best_ours:
        print(f"  {'Layer':>20}  {best_arditi['layer']:>{col_w}}  {best_ours['layer']:>{col_w}}")
        print(f"  {'Baseline':>20}  {arditi_quant['baseline']:>{col_w}.2%}  {ours_quant['baseline']:>{col_w}.2%}")
        print(f"  {'Global ablation':>20}  {arditi_quant['global_ablation']:>{col_w}.2%}  {ours_quant['global_ablation']:>{col_w}.2%}")
        print(f"  {'Layer ablation':>20}  {arditi_quant['layer_ablation']:>{col_w}.2%}  {ours_quant['layer_ablation']:>{col_w}.2%}")
        print(f"  {'Induction (s=1)':>20}  {arditi_quant['induction']:>{col_w}.2%}  {ours_quant['induction']:>{col_w}.2%}")

    # Mini eyeball for both
    for name, directions, best_info in [("arditi→arditi", arditi_directions, best_arditi),
                                         ("ours→arditi", our_directions, best_ours)]:
        if not best_info:
            continue
        vec = directions[(best_info["layer"], POS)]
        direction = DirectionVector(vector=vec, layer=best_info["layer"], position_index=POS, score=0)
        prompts = arditi_eval[1][:NUM_EYEBALL_PROMPTS]

        print(f"\n{'─'*70}")
        print(f"MINI EYEBALL: {name} (addition @ L{direction.layer}, strength={STRENGTH})")
        print(f"{'─'*70}")

        baseline_texts = generate_with_hooks(
            framework.model, framework.tokenizer,
            framework.prompt_formatter, prompts,
            max_new_tokens=64
        )
        framework.intervention_applier.apply_direction_intervention(
            direction, "add", strength=STRENGTH, layers=[direction.layer]
        )
        try:
            induced_texts = generate_with_hooks(
                framework.model, framework.tokenizer,
                framework.prompt_formatter, prompts,
                max_new_tokens=64
            )
        finally:
            framework.intervention_applier.clear_interventions()

        for i, (prompt, base, induced) in enumerate(zip(prompts, baseline_texts, induced_texts)):
            print(f"\n  {i+1}. Q: {prompt[:80]}")
            print(f"     Baseline: {base[:150]}")
            print(f"     Induced:  {induced[:150]}")

    elapsed = time.time() - total_t0
    print(f"\n{'═'*80}")
    print(f"EXPERIMENT COMPLETE — {elapsed/60:.1f} min")
    print(f"{'═'*80}")


if __name__ == "__main__":
    main()
