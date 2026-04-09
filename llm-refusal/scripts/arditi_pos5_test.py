"""
Quick test: Arditi's data at their exact spec (L12, pos=-5) on Llama-3-8B.

Tests whether the pos=-1 vs pos=-5 mismatch explains our failure to replicate.

Usage:
  python llm-refusal/scripts/arditi_pos5_test.py 2>&1 | tee results/arditi_pos5_test.log
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
from concept import DEFAULT_REFUSAL_PHRASES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ARDITI_LAYER = 12
ARDITI_POS = -5
MAX_POSITIONS = 6  # covers -1 through -6 (beyond -5)
STRENGTH = 1.0
NUM_SWEEP = 20
NUM_EYEBALL = 5


def load_arditi_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "arditi_harmful_train.json")) as f:
        harmful = [e["instruction"] for e in json.load(f)]
    with open(os.path.join(data_dir, "arditi_harmless_train.json")) as f:
        harmless = [e["instruction"] for e in json.load(f)]
    train_pos = harmful[:200]
    train_neg = harmless[:200]  # balanced subsample
    eval_pos = harmful[200:]
    eval_neg = harmless[1000:1080]
    return (train_pos, train_neg), (eval_pos, eval_neg)


def load_our_data():
    from prompts import create_refusal_train_data, create_refusal_eval_data
    return create_refusal_train_data(), create_refusal_eval_data()


def classify_response(text):
    if len(text.strip()) < 10:
        return "GARBLED"
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if len(text) > 0 and ascii_chars / len(text) < 0.5:
        return "GARBLED"
    if len(text) > 30:
        for i in range(0, min(len(text) - 30, 100)):
            chunk = text[i:i+10]
            if text.count(chunk) >= 3 and len(chunk.strip()) > 3:
                return "GARBLED"
    lower = text.lower()
    for phrase in DEFAULT_REFUSAL_PHRASES:
        if phrase.lower() in lower:
            return "DETECTED"
    return "NORMAL"


def compute_directions(framework, train_pos, train_neg, max_positions):
    all_data = PromptData(
        train_pos + train_neg,
        [True] * len(train_pos) + [False] * len(train_neg)
    )
    train_data, _ = all_data.train_val_split()
    return framework.finder.direction_finder_method.compute_difference_vectors(
        train_data, max_positions=max_positions
    )


def test_direction(framework, directions, layer, pos, eval_neg, strength, label):
    """Add direction at (layer, pos) and classify responses."""
    key = (layer, pos)
    if key not in directions:
        print(f"  {label}: No direction at ({layer}, {pos})")
        return None

    vec = directions[key]
    direction = DirectionVector(vector=vec, layer=layer, position_index=pos, score=0)
    norm = t.norm(vec).item()

    prompts = eval_neg[:NUM_SWEEP]
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

    classes = [classify_response(t) for t in texts]
    n_det = classes.count("DETECTED")
    n_garb = classes.count("GARBLED")
    n_norm = classes.count("NORMAL")
    print(f"  {label}: ||dir||={norm:.2f}  Detected={n_det}/{NUM_SWEEP}  Garbled={n_garb}  Normal={n_norm}")
    return {"texts": texts, "classes": classes, "n_det": n_det, "n_garb": n_garb, "norm": norm}


def main():
    t0 = time.time()
    print("=" * 80)
    print("ARDITI POS=-5 REPLICATION TEST")
    print(f"Model: {MODEL}")
    print(f"Arditi spec: layer={ARDITI_LAYER}, pos={ARDITI_POS}")
    print(f"max_positions={MAX_POSITIONS} (extracting -1 through -{MAX_POSITIONS})")
    print("=" * 80)

    # Load data
    arditi_train, arditi_eval = load_arditi_data()
    our_train, our_eval = load_our_data()
    print(f"\nArditi train: {len(arditi_train[0])} pos + {len(arditi_train[1])} neg")
    print(f"Our train: {len(our_train[0])} pos + {len(our_train[1])} neg")

    # Load model
    framework = DirectionTestFramework(model_name=MODEL, concept="refusal")
    print(f"Assistant prefix tokens: {framework.prompt_formatter.assistant_prefix_tokens}")

    # Compute directions at multiple positions
    print(f"\n--- Computing Arditi directions (max_positions={MAX_POSITIONS}) ---")
    arditi_dirs = compute_directions(framework, arditi_train[0], arditi_train[1], MAX_POSITIONS)

    print(f"\n--- Computing our directions (max_positions={MAX_POSITIONS}) ---")
    our_dirs = compute_directions(framework, our_train[0], our_train[1], MAX_POSITIONS)

    # Report available directions
    print(f"\n--- Available direction keys ---")
    arditi_keys = sorted(arditi_dirs.keys())
    our_keys = sorted(our_dirs.keys())
    print(f"  Arditi: {arditi_keys}")
    print(f"  Ours:   {our_keys}")

    # Norm comparison at L12 across positions
    print(f"\n--- Direction norms at L{ARDITI_LAYER} across positions ---")
    print(f"  {'Pos':>5}  {'||arditi||':>12}  {'||ours||':>12}  {'cos':>8}")
    for pos in range(-1, -MAX_POSITIONS - 1, -1):
        key = (ARDITI_LAYER, pos)
        if key in arditi_dirs and key in our_dirs:
            a_norm = t.norm(arditi_dirs[key]).item()
            o_norm = t.norm(our_dirs[key]).item()
            cos = t.nn.functional.cosine_similarity(
                arditi_dirs[key].unsqueeze(0), our_dirs[key].unsqueeze(0)
            ).item()
            marker = " ◀ ARDITI" if pos == ARDITI_POS else ""
            print(f"  {pos:>5}  {a_norm:>12.2f}  {o_norm:>12.2f}  {cos:>8.4f}{marker}")

    # Test Arditi's exact spec: L12, pos=-5
    print(f"\n{'='*80}")
    print(f"INDUCTION TEST AT ARDITI'S SPEC: L{ARDITI_LAYER}, pos={ARDITI_POS}")
    print(f"{'='*80}")
    print(f"\nArditi data, Arditi eval (benign prompts):")
    arditi_result = test_direction(framework, arditi_dirs, ARDITI_LAYER, ARDITI_POS, arditi_eval[1], STRENGTH, "arditi→arditi")

    print(f"\nOur data, Arditi eval:")
    ours_result = test_direction(framework, our_dirs, ARDITI_LAYER, ARDITI_POS, arditi_eval[1], STRENGTH, "ours→arditi")

    # Compare with pos=-1 (what we used before)
    print(f"\n--- Comparison: pos=-1 vs pos=-5 at L{ARDITI_LAYER} ---")
    print(f"\nArditi data, pos=-1:")
    test_direction(framework, arditi_dirs, ARDITI_LAYER, -1, arditi_eval[1], STRENGTH, "arditi@pos-1")
    print(f"\nArditi data, pos=-5:")
    test_direction(framework, arditi_dirs, ARDITI_LAYER, ARDITI_POS, arditi_eval[1], STRENGTH, "arditi@pos-5")

    # Sweep all positions at L12 with Arditi data
    print(f"\n--- Position sweep at L{ARDITI_LAYER} with Arditi data ---")
    for pos in range(-1, -MAX_POSITIONS - 1, -1):
        test_direction(framework, arditi_dirs, ARDITI_LAYER, pos, arditi_eval[1], STRENGTH, f"L{ARDITI_LAYER}/pos{pos}")

    # Sweep all positions at L12 with our data
    print(f"\n--- Position sweep at L{ARDITI_LAYER} with our data ---")
    for pos in range(-1, -MAX_POSITIONS - 1, -1):
        test_direction(framework, our_dirs, ARDITI_LAYER, pos, arditi_eval[1], STRENGTH, f"L{ARDITI_LAYER}/pos{pos}")

    # Eyeball at Arditi's spec
    if arditi_result and arditi_result["n_det"] > 0:
        print(f"\n{'─'*70}")
        print(f"EYEBALL: arditi→arditi at L{ARDITI_LAYER}/pos{ARDITI_POS}")
        print(f"{'─'*70}")
        prompts = arditi_eval[1][:NUM_EYEBALL]
        for i, (p, txt, cls) in enumerate(zip(prompts, arditi_result["texts"][:NUM_EYEBALL], arditi_result["classes"][:NUM_EYEBALL])):
            print(f"\n  {i+1}. [{cls}] Q: {p[:80]}")
            print(f"     A: {txt[:150]}")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
