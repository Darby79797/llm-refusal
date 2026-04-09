"""
Behavioral layer sweep for sycophancy: bypasses LogOdds scoring entirely.
For each (layer, pos) candidate, adds the direction via single-layer addition,
generates responses, and checks behavioral detection.

Tests hypothesis: "good direction exists but LogOdds can't see it"
vs "no clean sycophancy direction at any layer"

Usage:
  python llm-refusal/scripts/sycophancy_behavioral_sweep.py 2>&1 | tee results/sycophancy_behavioral_sweep.log
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch as t
import logging

from framework import DirectionTestFramework
from datatypes import PromptData, DirectionVector
from generation import generate_with_hooks
from concept import get_concept

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", [6, 8, 10, 12, 14, 16, 18]),
    ("Qwen/Qwen2.5-3B-Instruct",   [8, 11, 14, 17, 20, 23]),
]
POSITIONS = [-1, -2, -4]
NUM_EVAL = 20  # prompts per condition
CONCEPT = "sycophancy"


def main():
    concept = get_concept(CONCEPT)
    detect = concept.detection_fn
    eval_pos, eval_neg = concept.eval_data_fn()
    # Use negative (anti-sycophantic) prompts for induction test
    # If direction works, adding it should make model sycophantic on these
    test_prompts = eval_neg[:NUM_EVAL]

    for model_name, layers in MODELS:
        t0 = time.time()
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"Layers: {layers}, Positions: {POSITIONS}")
        print(f"{'='*80}")

        framework = DirectionTestFramework(model_name=model_name, concept=CONCEPT)

        # Compute directions at all positions
        train_pos, train_neg = concept.train_data_fn()
        train_data = PromptData(
            train_pos + train_neg,
            [True] * len(train_pos) + [False] * len(train_neg)
        )
        train_data, val_data = train_data.train_val_split()

        max_positions = max(abs(p) for p in POSITIONS)
        directions = framework.finder.direction_finder_method.compute_difference_vectors(
            train_data, max_positions=max_positions
        )

        # Baseline: no intervention
        baseline_texts = generate_with_hooks(
            framework.model, framework.tokenizer,
            framework.prompt_formatter, test_prompts,
            max_new_tokens=80
        )
        baseline_det = sum(1 for txt in baseline_texts if detect(txt))
        print(f"\nBaseline sycophancy: {baseline_det}/{NUM_EVAL} ({baseline_det/NUM_EVAL:.0%})")

        # Sweep
        print(f"\n{'Layer':>6} {'Pos':>5} {'||dir||':>8} {'Detected':>10} {'Rate':>6}")
        print(f"{'─'*6} {'─'*5} {'─'*8} {'─'*10} {'─'*6}")

        best = None
        for layer in layers:
            for pos in POSITIONS:
                key = (layer, pos)
                if key not in directions:
                    continue

                vec = directions[key]
                direction = DirectionVector(vector=vec, layer=layer, position_index=pos, score=0)
                norm = t.norm(vec).item()

                framework.intervention_applier.apply_direction_intervention(
                    direction, "add", strength=1.0, layers=[layer]
                )
                try:
                    texts = generate_with_hooks(
                        framework.model, framework.tokenizer,
                        framework.prompt_formatter, test_prompts,
                        max_new_tokens=80
                    )
                finally:
                    framework.intervention_applier.clear_interventions()

                n_det = sum(1 for txt in texts if detect(txt))
                rate = n_det / NUM_EVAL
                marker = ""
                if best is None or n_det > best[0]:
                    best = (n_det, layer, pos, norm, texts)
                    marker = " ◀"

                print(f"{layer:>6} {pos:>5} {norm:>8.2f} {n_det:>10}/{NUM_EVAL} {rate:>5.0%}{marker}")

        # Show best result eyeball
        if best:
            n_det, layer, pos, norm, texts = best
            print(f"\nBest: L{layer}/pos{pos} — {n_det}/{NUM_EVAL} detected ({n_det/NUM_EVAL:.0%})")
            print(f"\nEyeball (first 5 at best layer):")
            for i in range(min(5, len(test_prompts))):
                det = detect(texts[i])
                print(f"\n  [{i+1}] det={det}")
                print(f"  Q: {test_prompts[i][:100]}")
                print(f"  A: {texts[i][:200]}")

        elapsed = time.time() - t0
        print(f"\n{model_name} done in {elapsed/60:.1f} min")

        # Free memory
        del framework
        import gc; gc.collect()
        if t.backends.mps.is_available():
            t.mps.empty_cache()


if __name__ == "__main__":
    main()
