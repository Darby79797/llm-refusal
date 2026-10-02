"""
Test Arditi direction addition using the raw (unnormalized) vector.

The unit-normalized sweep (Part 3a of arditi_replication.py) showed 0/20 induction
across all strengths 1-12. But the raw direction has norm ~90 — maybe the original
magnitude is what's needed to push activations into the refusal regime.

Tests raw addition at strengths [0.1, 0.3, 0.5, 1.0, 1.5, 2.0] (effective norms
~9 to ~180) across the layers that worked best for ablation (L20, L22, L24).
Also includes a few qualitative examples.

Usage:
  python llm-refusal/scripts/arditi_raw_addition.py 2>&1 | tee results/arditi_raw_addition.log
"""
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import json
import time
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

MODEL = "meta-llama/Llama-2-7b-chat-hf"
LAYERS = [20, 22, 24]
STRENGTHS = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
NUM_EVAL = 20
NUM_EYEBALL = 5


def load_arditi_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "arditi_harmful_train.json")) as f:
        harmful = [e["instruction"] for e in json.load(f)]
    with open(os.path.join(data_dir, "arditi_harmless_train.json")) as f:
        harmless = [e["instruction"] for e in json.load(f)]
    return harmful[:200], harmless[:1000], harmless[1000:1080]


# Arditi's exact phrase list (refusal_arditi_exact), via the concept's detector.
ARDITI_REFUSAL = get_concept("refusal_arditi_exact")


def classify_response(text: str) -> str:
    if is_garbled(text):
        return "GARBLED"
    if ARDITI_REFUSAL.detect(text):
        return "DETECTED"
    return "NORMAL"


def main():
    total_t0 = time.time()

    print("=" * 80)
    print("ARDITI RAW ADDITION TEST")
    print(f"Model: {MODEL}")
    print(f"Layers: {LAYERS}, Strengths: {STRENGTHS}")
    print("=" * 80)

    # Load data
    train_harmful, train_harmless, eval_harmless = load_arditi_data()
    prompts = eval_harmless[:NUM_EVAL]
    eyeball_prompts = eval_harmless[:NUM_EYEBALL]
    print(f"  Eval prompts: {NUM_EVAL} harmless, {NUM_EYEBALL} for eyeball")

    # Load model
    framework = DirectionTestFramework(model_name=MODEL, concept="refusal")

    # Compute directions (imbalanced, 200:1000, same as arditi_replication)
    print("\n--- Computing directions (200 harmful + 1000 harmless) ---")
    t0 = time.time()
    all_data = PromptData(
        train_harmful + train_harmless,
        [True] * len(train_harmful) + [False] * len(train_harmless)
    )
    train_data, _ = all_data.train_val_split()
    diff_vectors = framework.finder.direction_finder_method.compute_difference_vectors(
        train_data, max_positions=1
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Print norms
    print("\n  Direction norms:")
    for layer in LAYERS:
        key = (layer, -1)
        if key in diff_vectors:
            print(f"    L{layer}: ||dir|| = {t.norm(diff_vectors[key]).item():.2f}")

    # ── Strength × Layer sweep (raw addition) ────────────────────────────
    print(f"\n{'═'*80}")
    print("RAW ADDITION SWEEP")
    print(f"{'═'*80}")

    print(f"\n  {'Layer':>5} {'Str':>5} {'||add||':>8} {'Det':>5} {'Garb':>5} {'Norm':>5}")
    print(f"  {'─'*5} {'─'*5} {'─'*8} {'─'*5} {'─'*5} {'─'*5}")

    best_detected = 0
    best_combo = None

    for layer in LAYERS:
        key = (layer, -1)
        if key not in diff_vectors:
            continue
        vec = diff_vectors[key]
        norm = t.norm(vec).item()

        for strength in STRENGTHS:
            effective_norm = norm * strength
            direction = DirectionVector(vector=vec, layer=layer, position_index=-1, score=0)

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
            n_det = classifications.count("DETECTED")
            n_garb = classifications.count("GARBLED")
            n_norm = classifications.count("NORMAL")

            marker = ""
            if n_det > best_detected and n_garb <= NUM_EVAL // 4:
                best_detected = n_det
                best_combo = (layer, strength, effective_norm, n_det, n_garb, n_norm, texts)
                marker = " ◀ BEST"

            print(f"  L{layer:>3} {strength:>5.1f} {effective_norm:>8.1f} {n_det:>5} {n_garb:>5} {n_norm:>5}{marker}")

        print()

    # ── Best combo summary ────────────────────────────────────────────────
    if best_combo:
        layer, strength, eff_norm, n_det, n_garb, n_norm, texts = best_combo
        print(f"{'═'*80}")
        print(f"BEST: Layer {layer}, strength={strength}, ||add||={eff_norm:.1f}")
        print(f"  Detected={n_det}/{NUM_EVAL}, Garbled={n_garb}/{NUM_EVAL}, Normal={n_norm}/{NUM_EVAL}")
        print(f"{'═'*80}")
    else:
        print("No detections at any combo.")

    # ── Eyeball: baseline vs best combo ───────────────────────────────────
    print(f"\n{'═'*80}")
    print("EYEBALL COMPARISON")
    print(f"{'═'*80}")

    # Baseline
    print("\n  [Baseline — no intervention]")
    baseline_texts = generate_with_hooks(
        framework.model, framework.tokenizer,
        framework.prompt_formatter, eyeball_prompts,
        max_new_tokens=64
    )
    for i, (p, txt) in enumerate(zip(eyeball_prompts, baseline_texts)):
        print(f"  {i+1}. Q: {p[:80]}")
        print(f"     A: {txt[:200]}")

    # Best combo (or fallback to L24 strength=1.0)
    if best_combo:
        layer, strength = best_combo[0], best_combo[1]
    else:
        layer, strength = 24, 1.0

    vec = diff_vectors[(layer, -1)]
    direction = DirectionVector(vector=vec, layer=layer, position_index=-1, score=0)
    eff_norm = t.norm(vec).item() * strength

    print(f"\n  [Raw addition — L{layer}, strength={strength}, ||add||={eff_norm:.1f}]")
    framework.intervention_applier.apply_direction_intervention(
        direction, "add", strength=strength, layers=[layer]
    )
    try:
        induced_texts = generate_with_hooks(
            framework.model, framework.tokenizer,
            framework.prompt_formatter, eyeball_prompts,
            max_new_tokens=64
        )
    finally:
        framework.intervention_applier.clear_interventions()

    for i, (p, txt) in enumerate(zip(eyeball_prompts, induced_texts)):
        cls = classify_response(txt)
        print(f"  {i+1}. Q: {p[:80]}")
        print(f"     A: {txt[:200]}  [{cls}]")

    # Also show a mid-range strength for comparison
    mid_strength = 0.5
    eff_mid = t.norm(vec).item() * mid_strength
    print(f"\n  [Raw addition — L{layer}, strength={mid_strength}, ||add||={eff_mid:.1f}]")
    framework.intervention_applier.apply_direction_intervention(
        direction, "add", strength=mid_strength, layers=[layer]
    )
    try:
        mid_texts = generate_with_hooks(
            framework.model, framework.tokenizer,
            framework.prompt_formatter, eyeball_prompts,
            max_new_tokens=64
        )
    finally:
        framework.intervention_applier.clear_interventions()

    for i, (p, txt) in enumerate(zip(eyeball_prompts, mid_texts)):
        cls = classify_response(txt)
        print(f"  {i+1}. Q: {p[:80]}")
        print(f"     A: {txt[:200]}  [{cls}]")

    elapsed = time.time() - total_t0
    print(f"\n{'═'*80}")
    print(f"COMPLETE — {elapsed/60:.1f} min")
    print(f"{'═'*80}")


if __name__ == "__main__":
    main()
