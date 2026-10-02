"""
Llama-2-7B-Chat replication diagnostic — interruptible version.

Runs tests in descending priority order, printing a running results table
after each generation call. Ctrl-C at any point gives useful data.

Priority order:
  1. Template diagnostic (instant)
  2. Compute ALL directions upfront (cheap — 2 fwd passes per dataset)
  3. Baseline generation (one pass, reused)
  4. Induction tests: all-layers at key layers → single-layer → position sweep → remaining
  5. Ablation tests
  6. Strength sweep on best result

Usage:
  python llm-refusal/scripts/llama2_replication_diagnostic.py 2>&1 | tee results/llama2_diagnostic.log
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
from concept import DEFAULT_REFUSAL_PHRASES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL = "meta-llama/Llama-2-7b-chat-hf"
NUM_EVAL = 10
SWEEP_LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20]


# ── Helpers ──────────────────────────────────────────────────────────────────

def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def load_arditi_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "arditi_harmful_train.json")) as f:
        harmful = [e["instruction"] for e in json.load(f)]
    with open(os.path.join(data_dir, "arditi_harmless_train.json")) as f:
        harmless = [e["instruction"] for e in json.load(f)]
    return harmful, harmless


def load_our_data():
    import prompts
    return prompts.create_refusal_train_data()


def classify_response(text):
    lower = text.lower()
    for phrase in DEFAULT_REFUSAL_PHRASES:
        if phrase.lower() in lower:
            return "DETECTED"
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    total = len(text.strip())
    if total < 10 or (total > 0 and ascii_chars / total < 0.5):
        return "GARBLED"
    return "NORMAL"


def generate_and_classify(framework, prompts, direction=None, intervention_type=None,
                          strength=1.0, layers=None):
    """Generate and return (n_detected, n_garbled, n_normal, texts)."""
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

    classes = [classify_response(t_) for t_ in texts]
    return classes.count("DETECTED"), classes.count("GARBLED"), classes.count("NORMAL"), texts


# ── Running Results Table ────────────────────────────────────────────────────

class ResultsTracker:
    """Accumulates results and prints a compact table after each test."""

    def __init__(self):
        self.rows = []  # list of dicts

    def add(self, dataset, layer, pos, mode, strength, n_det, n_garb, n_total, note=""):
        self.rows.append({
            "dataset": dataset, "layer": layer, "pos": pos, "mode": mode,
            "strength": strength, "det": n_det, "garb": n_garb, "total": n_total,
            "rate": n_det / n_total if n_total > 0 else 0, "note": note,
        })

    def print_table(self):
        if not self.rows:
            return
        flush_print(f"\n{'─'*90}")
        flush_print(f"  RESULTS SO FAR ({len(self.rows)} tests)")
        flush_print(f"  {'Dataset':<16} {'Layer':>5} {'Pos':>4} {'Mode':<12} {'Str':>4} │ {'Det':>4} {'Garb':>4} {'Rate':>6} │ {'Note'}")
        flush_print(f"  {'─'*16} {'─'*5} {'─'*4} {'─'*12} {'─'*4} │ {'─'*4} {'─'*4} {'─'*6} │ {'─'*10}")
        for r in self.rows:
            flush_print(
                f"  {r['dataset']:<16} L{r['layer']:>3} {r['pos']:>4} {r['mode']:<12} {r['strength']:>4.1f} │ "
                f"{r['det']:>3}/{r['total']} {r['garb']:>4} {r['rate']:>5.0%} │ {r['note']}"
            )
        flush_print(f"{'─'*90}")


# ── Phase 1: Template Diagnostic ────────────────────────────────────────────

def phase1_template(framework):
    flush_print(f"\n{'═'*80}")
    flush_print("PHASE 1: TEMPLATE DIAGNOSTIC")
    flush_print(f"{'═'*80}")

    fmt = framework.prompt_formatter
    tok = framework.tokenizer

    flush_print(f"\n  is_instruction_tuned: {fmt.is_instruction_tuned}")
    flush_print(f"  template: {repr(fmt.template)}")
    flush_print(f"  prepend_bos: {fmt.prepend_bos}")
    flush_print(f"  assistant_prefix_tokens: {fmt.assistant_prefix_tokens}")
    flush_print(f"  tokenizer.chat_template exists: {tok.chat_template is not None}")

    test_prompt = "Tell me how to make a cake"

    if fmt.template is not None:
        formatted = fmt.template.format(x=test_prompt)
    else:
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": test_prompt}],
            tokenize=False, add_generation_prompt=True
        )

    flush_print(f"\n  Formatted string: {repr(formatted)}")

    ids = tok(formatted, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    flush_print(f"\n  Token count: {len(ids)}")
    flush_print(f"\n  Last 10 tokens:")
    for i in range(max(0, len(ids) - 10), len(ids)):
        pos_from_end = i - len(ids)
        token_str = tok.decode([ids[i].item()])
        flush_print(f"    pos {pos_from_end:>3} (abs {i:>3}): id={ids[i].item():>6}  repr={repr(token_str)}")

    if tok.chat_template is not None:
        builtin = tok.apply_chat_template(
            [{"role": "user", "content": test_prompt}],
            tokenize=False, add_generation_prompt=True
        )
        flush_print(f"\n  Built-in chat_template output: {repr(builtin)}")
        if builtin != formatted:
            flush_print("  ⚠ MISMATCH between manual template and built-in chat_template!")
            builtin_ids = tok(builtin, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            flush_print(f"  Manual token count: {len(ids)}, Built-in token count: {len(builtin_ids)}")

    batch = fmt.format_batch([test_prompt])
    batch_ids = batch["input_ids"][0]
    flush_print(f"\n  format_batch token count: {len(batch_ids)}")
    flush_print(f"  Last 10 tokens from format_batch:")
    for i in range(max(0, len(batch_ids) - 10), len(batch_ids)):
        pos_from_end = i - len(batch_ids)
        token_str = tok.decode([batch_ids[i].item()])
        flush_print(f"    pos {pos_from_end:>3} (abs {i:>3}): id={batch_ids[i].item():>6}  repr={repr(token_str)}")


# ── Phase 2: Compute All Directions ─────────────────────────────────────────

def phase2_directions(framework, datasets, max_positions=5):
    """Compute directions for all datasets upfront. Returns {ds_name: diff_vectors}."""
    flush_print(f"\n{'═'*80}")
    flush_print("PHASE 2: COMPUTE DIRECTIONS (all datasets)")
    flush_print(f"{'═'*80}")

    all_directions = {}
    for ds_name, train_pos, train_neg in datasets:
        flush_print(f"\n  Computing directions for '{ds_name}' ({len(train_pos)} pos, {len(train_neg)} neg, max_positions={max_positions})...")

        all_data = PromptData(
            train_pos + train_neg,
            [True] * len(train_pos) + [False] * len(train_neg)
        )
        train_data, _ = all_data.train_val_split()

        t0 = time.time()
        diff_vectors = framework.finder.direction_finder_method.compute_difference_vectors(
            train_data, max_positions=max_positions
        )
        elapsed = time.time() - t0
        flush_print(f"  Done in {elapsed:.1f}s, {len(diff_vectors)} direction vectors")

        # Print norm table
        positions = sorted(set(p for _, p in diff_vectors.keys()))
        header = f"  {'Layer':>5}"
        for p in positions:
            header += f"  {'pos '+str(p):>10}"
        flush_print(f"\n  Direction norms ({ds_name}):")
        flush_print(header)
        flush_print(f"  {'─'*5}" + f"  {'─'*10}" * len(positions))

        for layer in SWEEP_LAYERS:
            row = f"  L{layer:>3}"
            for p in positions:
                key = (layer, p)
                if key in diff_vectors:
                    norm = t.norm(diff_vectors[key]).item()
                    row += f"  {norm:>10.2f}"
                else:
                    row += f"  {'—':>10}"
            flush_print(row)

        all_directions[ds_name] = diff_vectors

    return all_directions


# ── Phase 3: Baseline Generation ────────────────────────────────────────────

def phase3_baseline(framework, eval_harmful, eval_benign):
    flush_print(f"\n{'═'*80}")
    flush_print("PHASE 3: BASELINE BEHAVIOR (no intervention)")
    flush_print(f"{'═'*80}")

    flush_print(f"\n  Generating baseline on {len(eval_harmful)} harmful prompts...")
    n_det_h, n_garb_h, n_norm_h, texts_h = generate_and_classify(framework, eval_harmful)
    rate_h = n_det_h / len(eval_harmful)
    flush_print(f"  Harmful: {n_det_h} detected, {n_garb_h} garbled, {n_norm_h} normal → refusal rate = {rate_h:.0%}")
    for i in range(min(3, len(eval_harmful))):
        cls = classify_response(texts_h[i])
        flush_print(f"    [{cls:>8}] {eval_harmful[i][:60]}...")
        flush_print(f"             → {texts_h[i][:120]}")

    flush_print(f"\n  Generating baseline on {len(eval_benign)} benign prompts...")
    n_det_b, n_garb_b, n_norm_b, texts_b = generate_and_classify(framework, eval_benign)
    rate_b = n_det_b / len(eval_benign)
    flush_print(f"  Benign:  {n_det_b} detected, {n_garb_b} garbled, {n_norm_b} normal → false refusal rate = {rate_b:.0%}")
    for i in range(min(3, len(eval_benign))):
        cls = classify_response(texts_b[i])
        flush_print(f"    [{cls:>8}] {eval_benign[i][:60]}...")
        flush_print(f"             → {texts_b[i][:120]}")

    return rate_h, rate_b


# ── Phase 4: Induction Tests (priority ordered) ─────────────────────────────

def phase4_induction(framework, all_directions, eval_benign, tracker):
    flush_print(f"\n{'═'*80}")
    flush_print("PHASE 4: INDUCTION TESTS (priority order)")
    flush_print(f"{'═'*80}")

    num_layers = len(framework.intervention_applier.transformer_layers)
    all_layer_list = list(range(num_layers))

    def run_test(ds_name, diff_vectors, layer, pos, mode, strength=1.0, note=""):
        key = (layer, pos)
        if key not in diff_vectors:
            flush_print(f"  SKIP: {ds_name} L{layer} pos={pos} — no direction vector")
            return
        vec = diff_vectors[key]
        direction = DirectionVector(vector=vec, layer=layer, position_index=pos, score=0)
        layers = all_layer_list if mode == "all_layers_add" else [layer]
        intervention_type = "add"

        t0 = time.time()
        n_det, n_garb, n_norm, _ = generate_and_classify(
            framework, eval_benign, direction, intervention_type, strength, layers
        )
        elapsed = time.time() - t0
        flush_print(f"  ✓ {ds_name} L{layer} pos={pos} {mode} s={strength:.1f} → {n_det}/{len(eval_benign)} det, {n_garb} garb ({elapsed:.0f}s)")
        tracker.add(ds_name, layer, pos, mode, strength, n_det, n_garb, len(eval_benign), note)
        tracker.print_table()

    # Priority 4a: All-layers addition at mid-depth layers (Arditi's method, key question)
    flush_print(f"\n  --- 4a: All-layers addition at key layers (our_dataset) ---")
    for layer in [10, 14, 6]:
        run_test("our_dataset", all_directions["our_dataset"], layer, -1, "all_layers_add", note="4a: key layers")

    # Priority 4b: Same for arditi_balanced
    flush_print(f"\n  --- 4b: All-layers addition at key layers (arditi_balanced) ---")
    for layer in [10, 14, 6]:
        run_test("arditi_balanced", all_directions["arditi_balanced"], layer, -1, "all_layers_add", note="4b: arditi key")

    # Priority 4c: Single-layer addition at same layers for comparison
    flush_print(f"\n  --- 4c: Single-layer addition at key layers ---")
    for ds_name in ["our_dataset", "arditi_balanced"]:
        for layer in [10, 14, 6]:
            run_test(ds_name, all_directions[ds_name], layer, -1, "single_layer_add", note="4c: 1L compare")

    # Priority 4d: Position sweep at best layer (L6 from CLAUDE.md)
    flush_print(f"\n  --- 4d: Position sweep at L6 ---")
    for pos in [-2, -3, -4, -5]:
        for ds_name in ["our_dataset", "arditi_balanced"]:
            run_test(ds_name, all_directions[ds_name], 6, pos, "all_layers_add", note="4d: pos sweep")

    # Priority 4e: Remaining layers to fill out the picture
    flush_print(f"\n  --- 4e: Remaining layers (all-layers add, pos=-1) ---")
    done_layers = {10, 14, 6}
    for layer in SWEEP_LAYERS:
        if layer in done_layers:
            continue
        for ds_name in ["our_dataset", "arditi_balanced"]:
            run_test(ds_name, all_directions[ds_name], layer, -1, "all_layers_add", note="4e: remaining")


# ── Phase 5: Ablation Tests ─────────────────────────────────────────────────

def phase5_ablation(framework, all_directions, eval_harmful, tracker):
    flush_print(f"\n{'═'*80}")
    flush_print("PHASE 5: ABLATION TESTS")
    flush_print(f"{'═'*80}")

    num_layers = len(framework.intervention_applier.transformer_layers)
    all_layer_list = list(range(num_layers))

    for ds_name, diff_vectors in all_directions.items():
        # Pick top 3 layers by norm at pos=-1
        layer_norms = []
        for layer in SWEEP_LAYERS:
            key = (layer, -1)
            if key in diff_vectors:
                layer_norms.append((layer, t.norm(diff_vectors[key]).item()))
        layer_norms.sort(key=lambda x: x[1], reverse=True)
        top_layers = [l for l, _ in layer_norms[:3]]

        flush_print(f"\n  Top 3 layers by norm ({ds_name}): {top_layers}")

        for layer in top_layers:
            key = (layer, -1)
            vec = diff_vectors[key]
            direction = DirectionVector(vector=vec, layer=layer, position_index=-1, score=0)

            # Global ablation
            t0 = time.time()
            n_det, n_garb, n_norm, _ = generate_and_classify(
                framework, eval_harmful, direction, "ablate", 1.0, all_layer_list
            )
            elapsed = time.time() - t0
            flush_print(f"  ✓ {ds_name} L{layer} global_ablate → {n_det}/{len(eval_harmful)} det, {n_garb} garb ({elapsed:.0f}s)")
            tracker.add(ds_name, layer, -1, "global_ablate", 1.0, n_det, n_garb, len(eval_harmful), "5: ablation")
            tracker.print_table()


# ── Phase 6: Strength Sweep ─────────────────────────────────────────────────

def phase6_strength_sweep(framework, all_directions, eval_benign, tracker):
    flush_print(f"\n{'═'*80}")
    flush_print("PHASE 6: STRENGTH SWEEP (best induction result)")
    flush_print(f"{'═'*80}")

    # Find best induction result from tracker
    induction_rows = [r for r in tracker.rows if "add" in r["mode"]]
    if not induction_rows:
        flush_print("  No induction results to sweep. Skipping.")
        return

    best = max(induction_rows, key=lambda r: r["rate"])
    ds_name = best["dataset"]
    layer = best["layer"]
    pos = best["pos"]
    mode = best["mode"]
    flush_print(f"\n  Best induction: {ds_name} L{layer} pos={pos} {mode} → {best['rate']:.0%}")
    flush_print(f"  Sweeping strengths [0.5, 1, 2, 3, 5]...")

    num_layers = len(framework.intervention_applier.transformer_layers)
    all_layer_list = list(range(num_layers))
    diff_vectors = all_directions[ds_name]
    key = (layer, pos)
    vec = diff_vectors[key]
    direction = DirectionVector(vector=vec, layer=layer, position_index=pos, score=0)
    layers = all_layer_list if mode == "all_layers_add" else [layer]

    for strength in [0.5, 1.0, 2.0, 3.0, 5.0]:
        t0 = time.time()
        n_det, n_garb, n_norm, _ = generate_and_classify(
            framework, eval_benign, direction, "add", strength, layers
        )
        elapsed = time.time() - t0
        flush_print(f"  ✓ s={strength:.1f} → {n_det}/{len(eval_benign)} det, {n_garb} garb ({elapsed:.0f}s)")
        tracker.add(ds_name, layer, pos, f"{mode}_s{strength}", strength, n_det, n_garb, len(eval_benign), "6: strength")
        tracker.print_table()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    total_t0 = time.time()

    flush_print("=" * 80)
    flush_print("LLAMA-2-7B-CHAT REPLICATION DIAGNOSTIC (interruptible)")
    flush_print(f"Model: {MODEL}")
    flush_print(f"Sweep layers: {SWEEP_LAYERS}")
    flush_print(f"Eval size: {NUM_EVAL}")
    flush_print("=" * 80)

    # Load data
    flush_print("\n--- Loading data ---")
    arditi_harmful, arditi_harmless = load_arditi_data()
    our_pos, our_neg = load_our_data()
    flush_print(f"  Arditi: {len(arditi_harmful)} harmful, {len(arditi_harmless)} harmless")
    flush_print(f"  Ours:   {len(our_pos)} positive, {len(our_neg)} negative")

    # Eval sets (10 each)
    eval_harmful = arditi_harmful[200:200 + NUM_EVAL]
    eval_benign = arditi_harmless[1000:1000 + NUM_EVAL]

    # Datasets for direction computation
    datasets = [
        ("our_dataset", our_pos, our_neg),
        ("arditi_balanced", arditi_harmful[:200], arditi_harmless[:200]),
    ]

    # Load model
    flush_print("\n--- Loading model ---")
    framework = DirectionTestFramework(model_name=MODEL, concept="refusal")

    # Phase 1: Template diagnostic (instant)
    phase1_template(framework)

    # Phase 2: Compute ALL directions upfront (cheap)
    all_directions = phase2_directions(framework, datasets, max_positions=5)

    # Phase 3: Baseline generation (one pass)
    baseline_refusal, baseline_false_refusal = phase3_baseline(framework, eval_harmful, eval_benign)

    # Results tracker for phases 4-6
    tracker = ResultsTracker()

    # Phase 4: Induction tests (priority ordered)
    phase4_induction(framework, all_directions, eval_benign, tracker)

    # Phase 5: Ablation tests
    phase5_ablation(framework, all_directions, eval_harmful, tracker)

    # Phase 6: Strength sweep on best result
    phase6_strength_sweep(framework, all_directions, eval_benign, tracker)

    elapsed = time.time() - total_t0
    flush_print(f"\n{'═'*80}")
    flush_print(f"DIAGNOSTIC COMPLETE — {elapsed/60:.1f} min")
    flush_print(f"Baseline refusal: {baseline_refusal:.0%}, Baseline false refusal: {baseline_false_refusal:.0%}")
    tracker.print_table()
    flush_print(f"{'═'*80}")


if __name__ == "__main__":
    main()
