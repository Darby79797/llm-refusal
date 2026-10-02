"""
Arditi direction replication on Llama-2-7b-chat.

Tests three fixes for the Arditi direction's failure to induce refusal:
  1. Balanced subsampling (200:200 instead of 200:1000)
  2. Ablation-first evaluation (scale-invariant)
  3. Unit-normalized addition with strength/position sweeps

The Arditi dataset (260 AdvBench + 18793 Alpaca) produces directions with norms 73-79,
which destroy generation when added raw. Ablation (projection removal) is scale-invariant,
so we test that first. If ablation works, we know the direction carries refusal signal
and just needs proper scaling for addition.

Llama-2 template note: [INST] {x} [/INST] puts 4 template tokens after content.
Position -1 is ']', position -5 is the last content token.

Usage:
  python llm-refusal/scripts/arditi_replication.py 2>&1 | tee results/arditi_replication.log
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import json
import time
import random
import torch as t
import torch.nn.functional as F
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

# ── Configuration ────────────────────────────────────────────────────────────

MODEL = "meta-llama/Llama-2-7b-chat-hf"
LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
POSITIONS = [-1, -3, -5]
STRENGTHS = [1, 3, 5, 8, 12]
BALANCED_SEEDS = [42, 123, 456, 789, 1001]
NUM_EVAL = 20


# ── Data loading ─────────────────────────────────────────────────────────────

def load_arditi_data():
    """Load and split Arditi JSON data.

    Train: first 200 harmful + N harmless (N varies by config)
    Eval:  harmful[200:] (60) + harmless[1000:1080] (80)
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "arditi_harmful_train.json")) as f:
        harmful_raw = json.load(f)
    with open(os.path.join(data_dir, "arditi_harmless_train.json")) as f:
        harmless_raw = json.load(f)

    harmful = [entry["instruction"] for entry in harmful_raw]
    harmless = [entry["instruction"] for entry in harmless_raw]

    train_harmful = harmful[:200]
    eval_harmful = harmful[200:]        # 60
    eval_harmless = harmless[1000:1080]  # 80

    return train_harmful, harmless, eval_harmful, eval_harmless


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def compute_directions(framework, train_pos, train_neg, max_positions=1):
    """Compute diff-in-means direction vectors."""
    all_data = PromptData(
        train_pos + train_neg,
        [True] * len(train_pos) + [False] * len(train_neg)
    )
    train_data, _ = all_data.train_val_split()
    diff_vectors = framework.finder.direction_finder_method.compute_difference_vectors(
        train_data, max_positions=max_positions
    )
    return diff_vectors


def measure_refusal_rate(framework, prompts, direction=None, intervention_type=None,
                         strength=1.0, layers=None):
    """Generate responses and measure refusal detection rate.

    If direction is None, generates baseline (no intervention).
    Returns (rate, n_detected, n_garbled, n_normal, texts).
    """
    if direction is not None:
        framework.intervention_applier.apply_direction_intervention(
            direction, intervention_type, strength=strength, layers=layers
        )
    try:
        texts = generate_with_hooks(
            framework.model, framework.tokenizer,
            framework.prompt_formatter, prompts,
            max_new_tokens=64
        )
    finally:
        if direction is not None:
            framework.intervention_applier.clear_interventions()

    classifications = [classify_response(txt) for txt in texts]
    n_det = classifications.count("DETECTED")
    n_garb = classifications.count("GARBLED")
    n_norm = classifications.count("NORMAL")
    rate = n_det / len(prompts) if prompts else 0
    return rate, n_det, n_garb, n_norm, texts


# ── Part 1: Direction Diagnostics ────────────────────────────────────────────

def run_part1(framework, train_harmful, harmless):
    """Compute directions for 3 configs; report norms, pairwise cosines."""
    print(f"\n{'═'*80}")
    print("PART 1: DIRECTION DIAGNOSTICS")
    print(f"{'═'*80}")

    # Config 1: Imbalanced (200 harmful + 1000 harmless)
    print("\n  Computing imbalanced directions (200:1000)...")
    t0 = time.time()
    imb_dirs = compute_directions(framework, train_harmful, harmless[:1000])
    print(f"  Done in {time.time() - t0:.1f}s")

    # Config 2: Balanced (200 harmful + 200 harmless, seed=42)
    print("\n  Computing balanced directions (200:200, seed=42)...")
    t0 = time.time()
    rng = random.Random(42)
    harmless_sample = rng.sample(harmless[:1000], 200)
    bal_dirs = compute_directions(framework, train_harmful, harmless_sample)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Config 3: Balanced-averaged (average of 5 seeds, unit-normalized each)
    print(f"\n  Computing balanced-averaged directions ({len(BALANCED_SEEDS)} seeds)...")
    t0 = time.time()
    seed_dirs_list = []
    for seed in BALANCED_SEEDS:
        rng = random.Random(seed)
        sample = rng.sample(harmless[:1000], 200)
        seed_dirs = compute_directions(framework, train_harmful, sample)
        seed_dirs_list.append(seed_dirs)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Average: unit-normalize each seed's direction, average, re-normalize
    avg_dirs = {}
    for key in imb_dirs:
        vecs = []
        for sd in seed_dirs_list:
            if key in sd:
                v = sd[key]
                vecs.append(v / t.norm(v))
        if vecs:
            avg = t.stack(vecs).mean(dim=0)
            avg = avg / t.norm(avg)  # re-normalize
            avg_dirs[key] = avg

    # Report norms
    configs = [
        ("imbalanced", imb_dirs),
        ("balanced", bal_dirs),
        ("balanced-avg", avg_dirs),
    ]

    print(f"\n  {'Layer':>5}", end="")
    for name, _ in configs:
        print(f"  {name:>14}", end="")
    print()
    print(f"  {'─'*5}", end="")
    for _ in configs:
        print(f"  {'─'*14}", end="")
    print()

    for layer in LAYERS:
        key = (layer, -1)
        print(f"  L{layer:>3}", end="")
        for _, dirs in configs:
            if key in dirs:
                norm = t.norm(dirs[key]).item()
                print(f"  {norm:>14.2f}", end="")
            else:
                print(f"  {'N/A':>14}", end="")
        print()

    # Pairwise cosine similarities between configs
    print(f"\n  Pairwise cosine similarities (per layer):")
    pairs = [
        ("imb↔bal", imb_dirs, bal_dirs),
        ("imb↔avg", imb_dirs, avg_dirs),
        ("bal↔avg", bal_dirs, avg_dirs),
    ]
    print(f"  {'Layer':>5}", end="")
    for name, _, _ in pairs:
        print(f"  {name:>10}", end="")
    print()
    print(f"  {'─'*5}", end="")
    for _ in pairs:
        print(f"  {'─'*10}", end="")
    print()

    for layer in LAYERS:
        key = (layer, -1)
        print(f"  L{layer:>3}", end="")
        for _, d1, d2 in pairs:
            if key in d1 and key in d2:
                cos = F.cosine_similarity(d1[key].unsqueeze(0).float(),
                                          d2[key].unsqueeze(0).float()).item()
                print(f"  {cos:>10.4f}", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()

    # Cosine to saved direction if it exists
    saved_path = os.path.join("results", "Llama-2-7b-chat-hf-refusal-direction")
    if os.path.exists(f"{saved_path}.pt"):
        saved = DirectionVector.load(saved_path)
        print(f"\n  Cosine to saved direction (L{saved.layer}, P{saved.position_index}):")
        for name, dirs in configs:
            key = (saved.layer, saved.position_index)
            if key in dirs:
                cos = F.cosine_similarity(dirs[key].unsqueeze(0).float(),
                                          saved.vector.unsqueeze(0).float()).item()
                print(f"    {name}: {cos:.4f}")

    return configs


# ── Part 2: Ablation Evaluation ──────────────────────────────────────────────

def pick_top_layers(configs, n=3):
    """Pick top N layers by norm from imbalanced config (they're similar across configs)."""
    imb_dirs = configs[0][1]
    layer_norms = []
    for layer in LAYERS:
        key = (layer, -1)
        if key in imb_dirs:
            layer_norms.append((layer, t.norm(imb_dirs[key]).item()))
    layer_norms.sort(key=lambda x: x[1], reverse=True)
    return [l for l, _ in layer_norms[:n]]


def run_part2(framework, configs, eval_harmful):
    """Ablation eval for imbalanced and balanced directions at top layers."""
    print(f"\n{'═'*80}")
    print("PART 2: ABLATION EVALUATION")
    print(f"{'═'*80}")

    num_layers = len(framework.intervention_applier.transformer_layers)
    all_layers = list(range(num_layers))
    top_layers = pick_top_layers(configs)
    print(f"\n  Top layers by norm: {top_layers}")
    print(f"  Eval prompts: {len(eval_harmful)} harmful")

    # Baseline refusal rate (no intervention)
    print("\n  Measuring baseline refusal rate...")
    base_rate, base_det, _, _, _ = measure_refusal_rate(framework, eval_harmful)
    print(f"  Baseline: {base_det}/{len(eval_harmful)} = {base_rate:.1%}")

    # Test imbalanced and balanced
    test_configs = [
        ("imbalanced", configs[0][1]),
        ("balanced", configs[1][1]),
    ]

    results = {}
    for config_name, dirs in test_configs:
        print(f"\n  --- {config_name} ---")
        results[config_name] = {"baseline": base_rate}

        for layer in top_layers:
            key = (layer, -1)
            if key not in dirs:
                continue
            vec = dirs[key]
            direction = DirectionVector(vector=vec, layer=layer, position_index=-1, score=0)

            # Global ablation
            rate, n_det, n_garb, _, _ = measure_refusal_rate(
                framework, eval_harmful, direction, "ablate", 1.0, all_layers
            )
            delta = rate - base_rate
            print(f"  L{layer:>2} global ablation: {n_det}/{len(eval_harmful)} = {rate:.1%} (Δ = {delta:+.1%})")

            # Layer-specific ablation
            rate_l, n_det_l, _, _, _ = measure_refusal_rate(
                framework, eval_harmful, direction, "ablate", 1.0, [layer]
            )
            delta_l = rate_l - base_rate
            print(f"  L{layer:>2} layer  ablation: {n_det_l}/{len(eval_harmful)} = {rate_l:.1%} (Δ = {delta_l:+.1%})")

            if config_name not in results:
                results[config_name] = {}
            results[config_name][f"L{layer}_global"] = rate
            results[config_name][f"L{layer}_layer"] = rate_l

    return results, top_layers


# ── Part 3: Unit-Normalized Addition Sweep ───────────────────────────────────

def run_part3(framework, configs, part2_results, top_layers, eval_harmless,
              train_harmful, harmless):
    """Unit-normalized addition: strength sweep + position sweep."""
    print(f"\n{'═'*80}")
    print("PART 3: UNIT-NORMALIZED ADDITION SWEEP")
    print(f"{'═'*80}")

    # Determine which configs showed good ablation (>20pp drop)
    base_rate = part2_results.get("imbalanced", {}).get("baseline", 1.0)
    good_configs = []
    for config_name in ["imbalanced", "balanced"]:
        res = part2_results.get(config_name, {})
        for layer in top_layers:
            global_key = f"L{layer}_global"
            if global_key in res:
                delta = base_rate - res[global_key]  # drop in refusal
                if delta > 0.20:
                    good_configs.append((config_name, layer))

    if not good_configs:
        print("\n  No configs showed >20pp ablation drop. Testing all anyway.")
        good_configs = [(name, top_layers[0]) for name, _ in [("imbalanced", None), ("balanced", None)]]

    prompts = eval_harmless[:NUM_EVAL]
    print(f"  Eval prompts: {len(prompts)} harmless")

    # ── 3a: Strength sweep at position -1 ──
    print(f"\n  --- 3a: Strength sweep (position -1) ---")

    config_map = {name: dirs for name, dirs in configs}

    summary_3a = {}
    for config_name, best_layer in good_configs:
        dirs = config_map[config_name]
        key = (best_layer, -1)
        if key not in dirs:
            continue

        vec = dirs[key]
        unit_vec = vec / t.norm(vec)
        direction = DirectionVector(vector=unit_vec, layer=best_layer, position_index=-1, score=0)

        print(f"\n  {config_name} @ L{best_layer} (unit-normalized):")
        print(f"  {'Strength':>8} {'Detected':>9} {'Garbled':>8} {'Normal':>7}")
        print(f"  {'─'*8} {'─'*9} {'─'*8} {'─'*7}")

        best_strength = None
        best_det = -1
        for s in STRENGTHS:
            rate, n_det, n_garb, n_norm, _ = measure_refusal_rate(
                framework, prompts, direction, "add", s, [best_layer]
            )
            marker = ""
            if n_det > best_det and n_garb <= len(prompts) // 4:
                best_det = n_det
                best_strength = s
                marker = " ◀"
            print(f"  {s:>8} {n_det:>9} {n_garb:>8} {n_norm:>7}{marker}")

        summary_3a[(config_name, best_layer)] = best_strength
        print(f"  Best strength: {best_strength}")

    # ── 3b: Position sweep at best strength ──
    print(f"\n  --- 3b: Position sweep ---")

    # Need directions at multiple positions
    # Pick the first good config for position sweep
    if good_configs:
        config_name, best_layer = good_configs[0]
        best_s = summary_3a.get((config_name, best_layer), STRENGTHS[2])

        dirs_config = config_map[config_name]
        # Get train data for this config
        if config_name == "imbalanced":
            train_neg = harmless[:1000]
        else:
            rng = random.Random(42)
            train_neg = rng.sample(harmless[:1000], 200)

        print(f"\n  Recomputing directions at max_positions=5 for {config_name}...")
        t0 = time.time()
        multi_pos_dirs = compute_directions(framework, train_harmful, train_neg, max_positions=5)
        print(f"  Done in {time.time() - t0:.1f}s")

        # Show what token each position corresponds to
        sample_prompt = eval_harmless[0]
        template = framework.prompt_formatter.template
        if template is not None:
            formatted_str = template.format(x=sample_prompt)
        else:
            formatted_str = framework.tokenizer.apply_chat_template(
                [{"role": "user", "content": sample_prompt}],
                tokenize=False, add_generation_prompt=True
            )
        input_ids = framework.tokenizer(formatted_str, return_tensors="pt")["input_ids"][0]
        print(f"\n  Token positions for sample prompt:")
        for pos in POSITIONS:
            token_id = input_ids[pos].item()
            token_str = framework.tokenizer.decode([token_id])
            print(f"    pos {pos}: token_id={token_id}, repr={repr(token_str)}")

        print(f"\n  Position sweep @ {config_name}, L{best_layer}, strength={best_s}:")
        print(f"  {'Position':>8} {'Detected':>9} {'Garbled':>8} {'Normal':>7} {'Norm':>8}")
        print(f"  {'─'*8} {'─'*9} {'─'*8} {'─'*7} {'─'*8}")

        for pos in POSITIONS:
            key = (best_layer, pos)
            if key not in multi_pos_dirs:
                print(f"  {pos:>8} {'N/A':>9}")
                continue

            vec = multi_pos_dirs[key]
            norm = t.norm(vec).item()
            unit_vec = vec / t.norm(vec)
            direction = DirectionVector(vector=unit_vec, layer=best_layer,
                                        position_index=pos, score=0)

            rate, n_det, n_garb, n_norm, _ = measure_refusal_rate(
                framework, prompts, direction, "add", best_s, [best_layer]
            )
            print(f"  {pos:>8} {n_det:>9} {n_garb:>8} {n_norm:>7} {norm:>8.2f}")

    return summary_3a


# ── Part 4: Summary Table ────────────────────────────────────────────────────

def run_part4(configs, part2_results, top_layers, summary_3a):
    """Print final summary table."""
    print(f"\n{'═'*80}")
    print("PART 4: SUMMARY")
    print(f"{'═'*80}")

    base_rate = part2_results.get("imbalanced", {}).get("baseline", None)
    print(f"\n  Baseline refusal rate: {base_rate:.1%}" if base_rate else "")

    config_map = {name: dirs for name, dirs in configs}

    print(f"\n  {'Config':<16} {'Layer':>5} {'Norm':>8} {'Abl Δ':>8} {'Best s':>7} {'Notes'}")
    print(f"  {'─'*16} {'─'*5} {'─'*8} {'─'*8} {'─'*7} {'─'*20}")

    for config_name, dirs in configs:
        for layer in top_layers[:1]:  # just top layer
            key = (layer, -1)
            if key not in dirs:
                continue
            norm = t.norm(dirs[key]).item()

            # Ablation delta
            res = part2_results.get(config_name, {})
            global_key = f"L{layer}_global"
            if global_key in res and base_rate is not None:
                abl_delta = res[global_key] - base_rate
                abl_str = f"{abl_delta:+.1%}"
            else:
                abl_str = "N/A"

            # Best strength
            best_s = summary_3a.get((config_name, layer), None)
            s_str = str(best_s) if best_s else "N/A"

            notes = ""
            if config_name == "balanced-avg":
                notes = "unit-avg of 5 seeds"

            print(f"  {config_name:<16} L{layer:>3} {norm:>8.2f} {abl_str:>8} {s_str:>7} {notes}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    total_t0 = time.time()

    print("=" * 80)
    print("ARDITI DIRECTION REPLICATION ON LLAMA-2-7B-CHAT")
    print(f"Model: {MODEL}")
    print(f"Layers: {LAYERS}")
    print(f"Positions: {POSITIONS}")
    print(f"Strengths: {STRENGTHS}")
    print("=" * 80)

    # Load data
    print("\n--- Loading data ---")
    train_harmful, harmless, eval_harmful, eval_harmless = load_arditi_data()
    print(f"  Train harmful: {len(train_harmful)}")
    print(f"  Total harmless pool: {len(harmless)}")
    print(f"  Eval harmful: {len(eval_harmful)}")
    print(f"  Eval harmless: {len(eval_harmless)}")

    # Load model
    print("\n--- Loading model ---")
    framework = DirectionTestFramework(model_name=MODEL, concept="refusal")

    # Part 1: Direction diagnostics
    configs = run_part1(framework, train_harmful, harmless)

    # Part 2: Ablation evaluation
    part2_results, top_layers = run_part2(framework, configs, eval_harmful)

    # Part 3: Unit-normalized addition sweep
    summary_3a = run_part3(framework, configs, part2_results, top_layers,
                           eval_harmless, train_harmful, harmless)

    # Part 4: Summary
    run_part4(configs, part2_results, top_layers, summary_3a)

    elapsed = time.time() - total_t0
    print(f"\n{'═'*80}")
    print(f"EXPERIMENT COMPLETE — {elapsed/60:.1f} min")
    print(f"{'═'*80}")


if __name__ == "__main__":
    main()
