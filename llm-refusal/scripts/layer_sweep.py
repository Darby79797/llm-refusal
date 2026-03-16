"""
Layer sweep: scan layers with raw vs unit normalization on negative eval prompts.
Classifies each response as: DETECTED, GARBLED, or NORMAL.

Usage:
  python llm-refusal/scripts/layer_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct
  python llm-refusal/scripts/layer_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct --concept sycophancy --layers 4,8,12,16
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import torch as t
import logging

from framework import DirectionTestFramework
from datatypes import PromptData, DirectionVector
from generation import generate_with_hooks
from concept import get_concept, ConceptDefinition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_LAYERS = [12, 13, 14, 15]
POS = -1
NUM_PROMPTS = 20


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


def classify_response(text: str, concept: ConceptDefinition) -> str:
    """Classify a response as DETECTED, GARBLED, or NORMAL using the concept's detection."""
    if is_garbled(text):
        return "GARBLED"
    # Use concept's detection_fn if available, else phrase matching
    if concept.detection_fn is not None:
        if concept.detection_fn(text):
            return "DETECTED"
    else:
        lower = text.lower()
        for phrase in concept.detection_phrases:
            if phrase.lower() in lower:
                return "DETECTED"
    return "NORMAL"


def run_sweep(model_name, layers=None, concept_name="refusal"):
    layers = layers or DEFAULT_LAYERS
    print(f"\n{'='*80}")
    print(f"LAYER SWEEP: {model_name} [{concept_name}]")
    print(f"Layers {layers}, pos={POS}, {NUM_PROMPTS} negative eval prompts")
    print(f"{'='*80}")

    t0 = time.time()

    # Load model and compute directions for all layers
    framework = DirectionTestFramework(model_name=model_name, concept=concept_name)
    concept = get_concept(concept_name)
    positive, negative = concept.train_data_fn()
    all_data = PromptData(
        positive + negative,
        [True] * len(positive) + [False] * len(negative)
    )
    train_data, val_data = all_data.train_val_split()

    logger.info("Computing difference-in-means vectors...")
    diff_vectors = framework.finder.direction_finder_method.compute_difference_vectors(
        train_data, max_positions=1
    )

    # Get benign eval prompts
    eval_pos, eval_neg = concept.eval_data_fn()
    benign_prompts = eval_neg[:NUM_PROMPTS]

    # Generate baseline
    logger.info("Generating baseline responses...")
    baseline_texts = generate_with_hooks(
        framework.model, framework.tokenizer,
        framework.prompt_formatter, benign_prompts,
        max_new_tokens=64
    )

    # Store all results
    results = {}

    for layer in layers:
        key = (layer, POS)
        if key not in diff_vectors:
            print(f"\n  WARNING: No direction vector for layer={layer}, pos={POS}")
            continue

        vec = diff_vectors[key]
        direction = DirectionVector(vector=vec, layer=layer, position_index=POS, score=0)
        norm = t.norm(direction.vector).item()

        for mode_label, use_raw in [("raw", True), ("unit", False)]:
            condition = f"L{layer}_{mode_label}"
            logger.info(f"Running {condition}...")

            # Apply intervention
            if use_raw:
                strength = 1.0
            else:
                strength = 1.0 / norm if norm > 0 else 1.0

            framework.intervention_applier.apply_direction_intervention(
                direction, "add", strength=strength, layers=[layer]
            )
            try:
                texts = generate_with_hooks(
                    framework.model, framework.tokenizer,
                    framework.prompt_formatter, benign_prompts,
                    max_new_tokens=64
                )
            finally:
                framework.intervention_applier.clear_interventions()

            # Classify
            classifications = [classify_response(txt, concept) for txt in texts]
            n_detected = classifications.count("DETECTED")
            n_garbled = classifications.count("GARBLED")
            n_normal = classifications.count("NORMAL")

            results[condition] = {
                "norm": norm,
                "texts": texts,
                "classifications": classifications,
                "n_detected": n_detected,
                "n_garbled": n_garbled,
                "n_normal": n_normal,
            }

    # ── Print summary table ──────────────────────────────────────────────
    det_label = concept_name.capitalize()
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name} [{concept_name}]")
    print(f"{'='*80}")
    print(f"  {'Condition':<16} {'||dir||':>8} {det_label:>8} {'Garbled':>8} {'Normal':>8} {'Effective ||add||':>18}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*18}")

    for layer in layers:
        for mode_label, use_raw in [("raw", True), ("unit", False)]:
            condition = f"L{layer}_{mode_label}"
            if condition not in results:
                continue
            r = results[condition]
            if use_raw:
                eff_mag = r["norm"]
            else:
                eff_mag = 1.0
            print(f"  {condition:<16} {r['norm']:>8.2f} {r['n_detected']:>8} {r['n_garbled']:>8} {r['n_normal']:>8} {eff_mag:>18.2f}")

    # ── Print detailed responses for each condition ──────────────────────
    for layer in layers:
        for mode_label, use_raw in [("raw", True), ("unit", False)]:
            condition = f"L{layer}_{mode_label}"
            if condition not in results:
                continue
            r = results[condition]
            print(f"\n{'─'*80}")
            print(f"DETAIL: {condition} (norm={r['norm']:.2f}, eff_add={'%.2f' % r['norm'] if use_raw else '1.00'})")
            print(f"  {det_label}={r['n_detected']}, Garbled={r['n_garbled']}, Normal={r['n_normal']}")
            print(f"{'─'*80}")
            for i, (prompt, text, cls) in enumerate(zip(benign_prompts, r["texts"], r["classifications"])):
                tag = f"[{cls}]"
                print(f"  {i+1:2d}. {tag:<10} Q: {prompt[:60]}")
                print(f"              A: {text[:150]}")
                print()

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: 12,13,14,15)")
    parser.add_argument("--concept", type=str, default="refusal",
                        help="Concept to sweep (default: refusal)")
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    run_sweep(args.model, layers=layers, concept_name=args.concept)
