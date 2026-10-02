"""
Diagnostic: sweep layer × strength on Llama-2-7b-chat to find the refusal sweet spot.

On Qwen models, direction norms at the sweet spot yield norm/√hidden_dim ≈ 0.2-0.55.
On Llama-2 at L8 (best layer from prior run), norm/√4096 = 0.061 — ~7-9x weaker.
This sweep tests whether higher strengths recover refusal induction.

Usage:
  python llm-refusal/scripts/llama2_strength_sweep.py 2>&1 | tee results/llama2_strength_sweep.log
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

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

MODEL = "meta-llama/Llama-2-7b-chat-hf"
LAYERS = [6, 8, 10, 12, 14, 16]
STRENGTHS = [1, 3, 5, 8]
POS = -1
NUM_PROMPTS = 20


def is_garbled(text: str) -> bool:
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
    if is_garbled(text):
        return "GARBLED"
    lower = text.lower()
    for phrase in DEFAULT_REFUSAL_PHRASES:
        if phrase.lower() in lower:
            return "DETECTED"
    return "NORMAL"


def main():
    t0 = time.time()
    from prompts import create_refusal_train_data, create_refusal_eval_data
    train_pos, train_neg = create_refusal_train_data()
    _, eval_neg = create_refusal_eval_data()
    prompts = eval_neg[:NUM_PROMPTS]

    framework = DirectionTestFramework(model_name=MODEL, concept="refusal")

    # Compute directions
    all_data = PromptData(
        train_pos + train_neg,
        [True] * len(train_pos) + [False] * len(train_neg)
    )
    train_data, _ = all_data.train_val_split()
    diff_vectors = framework.finder.direction_finder_method.compute_difference_vectors(
        train_data, max_positions=1
    )

    # Print direction norms at each layer
    print(f"\n{'═'*80}")
    print(f"DIRECTION NORMS (all layers)")
    print(f"{'═'*80}")
    for layer in LAYERS:
        key = (layer, POS)
        if key in diff_vectors:
            norm = t.norm(diff_vectors[key]).item()
            print(f"  L{layer:>2}: ||dir|| = {norm:.2f}")

    # Sweep layer × strength
    print(f"\n{'═'*80}")
    print(f"LAYER × STRENGTH SWEEP ({NUM_PROMPTS} benign prompts)")
    print(f"{'═'*80}")

    # Header
    print(f"\n  {'Layer':>5} {'Str':>4} {'||add||':>8} {'Det':>5} {'Garb':>5} {'Norm':>5}")
    print(f"  {'─'*5} {'─'*4} {'─'*8} {'─'*5} {'─'*5} {'─'*5}")

    best_detected = 0
    best_combo = None

    for layer in LAYERS:
        key = (layer, POS)
        if key not in diff_vectors:
            continue
        vec = diff_vectors[key]
        norm = t.norm(vec).item()

        for strength in STRENGTHS:
            effective_norm = norm * strength
            direction = DirectionVector(vector=vec, layer=layer, position_index=POS, score=0)

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
            if n_det > best_detected and n_garb <= 5:
                best_detected = n_det
                best_combo = (layer, strength, effective_norm, n_det, n_garb, n_norm)
                marker = " ◀ BEST"

            print(f"  L{layer:>3} {strength:>4} {effective_norm:>8.1f} {n_det:>5} {n_garb:>5} {n_norm:>5}{marker}")

        print()  # blank line between layers

    if best_combo:
        layer, strength, eff_norm, n_det, n_garb, n_norm = best_combo
        print(f"\n{'═'*80}")
        print(f"BEST COMBO: Layer {layer}, strength={strength}")
        print(f"  Effective ||add|| = {eff_norm:.1f}")
        print(f"  Detected={n_det}/{NUM_PROMPTS}, Garbled={n_garb}/{NUM_PROMPTS}, Normal={n_norm}/{NUM_PROMPTS}")
        print(f"{'═'*80}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
