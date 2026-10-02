"""
Refusal circuit analysis: identify which attention heads and MLP sublayers
write the refusal direction into the residual stream.

For each model:
  1. Layer-level attribution (contrastive: harmful - benign)
  2. Head-level decomposition on top layers
  3. Causal verification: ablate each component's refusal contribution, measure log-odds change
  4. Plots: layer attribution bars, head heatmaps, causal effect chart

Models tested in order of size (smallest first for fast iteration):
  - Qwen2.5-0.5B (24L, 14 heads)
  - Qwen2.5-1.5B (28L, 12 heads)
  - Qwen2.5-3B  (36L, 16 heads)
  - Llama-3-8B   (32L, 32 heads)
  - Qwen2.5-7B  (28L, 28 heads)

Usage:
  python llm-refusal/scripts/refusal_circuit.py 2>&1 | tee results/circuit/refusal_circuit.log
  python llm-refusal/scripts/refusal_circuit.py --models Qwen/Qwen2.5-0.5B-Instruct  # single model
"""
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import gc
import json
import argparse
import logging
import time
import numpy as np
import torch as t

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datatypes import DirectionVector
from formatting import ChatPromptFormatter
from interventions import ModelInterventionApplier
from attribution import AttributionAnalyzer, CircuitResult
from prompts import create_refusal_eval_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────

ALL_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

NUM_ATTRIBUTION_PROMPTS = 30   # prompts for attribution (1 forward pass)
NUM_CAUSAL_PROMPTS = 20        # prompts for causal verification (N forward passes)
TOP_LAYERS_K = 5               # layers to decompose into heads
TOP_COMPONENTS_K = 15          # components to verify causally

RESULTS_DIR = "results/circuit"
PLOTS_DIR = "plots/circuit"


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_layer_attributions(result: CircuitResult, save_dir: str):
    """Stacked bar chart: attn + MLP projection at each layer (contrastive)."""
    attrs = result.contrastive_layer_attributions
    layers = [a.layer_idx for a in attrs]
    attn_vals = [a.attn_projection for a in attrs]
    mlp_vals = [a.mlp_projection for a in attrs]

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.4), 5))

    x = np.arange(len(layers))
    width = 0.35

    bars_attn = ax.bar(x - width/2, attn_vals, width, label='Attention', color='#4C72B0', alpha=0.85)
    bars_mlp = ax.bar(x + width/2, mlp_vals, width, label='MLP', color='#DD8452', alpha=0.85)

    # Highlight the direction layer
    dir_layer = result.direction_layer
    if dir_layer in layers:
        idx = layers.index(dir_layer)
        ax.axvline(x=idx, color='red', linestyle='--', alpha=0.5, label=f'Direction layer (L{dir_layer})')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Contrastive projection onto refusal direction')
    ax.set_title(f'Layer Attribution (harmful - benign) — {result.model_name.split("/")[-1]}')
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(l) for l in layers[::2]])
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()

    short = result.model_name.split("/")[-1]
    path = os.path.join(save_dir, f"{short}-layer_attribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_head_heatmap(result: CircuitResult, save_dir: str, use_contrastive=True):
    """Heatmap of per-head attribution for the top layers."""
    attrs = result.contrastive_head_attributions if use_contrastive else result.head_attributions
    if not attrs:
        return None

    layers = [a.layer_idx for a in attrs]
    num_heads = max(len(a.head_projections) for a in attrs)

    # Build matrix: rows=layers, cols=heads, plus an extra column for MLP
    matrix = np.zeros((len(layers), num_heads + 1))
    for i, a in enumerate(attrs):
        for h in range(num_heads):
            matrix[i, h] = a.head_projections.get(h, 0.0)
        matrix[i, -1] = a.mlp_projection

    fig, ax = plt.subplots(figsize=(max(8, (num_heads + 1) * 0.5), max(3, len(layers) * 0.6)))

    vmax = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap='RdBu_r', norm=norm, aspect='auto')

    col_labels = [f'H{h}' for h in range(num_heads)] + ['MLP']
    ax.set_xticks(range(num_heads + 1))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f'L{l}' for l in layers])

    # Annotate cells with values for clarity
    for i in range(len(layers)):
        for j in range(num_heads + 1):
            val = matrix[i, j]
            if abs(val) > vmax * 0.15:  # only annotate significant cells
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if abs(val) > vmax * 0.6 else 'black')

    fig.colorbar(im, ax=ax, shrink=0.8)
    label = "Contrastive" if use_contrastive else "Harmful-only"
    ax.set_title(f'Head Attribution ({label}) — {result.model_name.split("/")[-1]}')
    plt.tight_layout()

    short = result.model_name.split("/")[-1]
    suffix = "contrastive" if use_contrastive else "positive"
    path = os.path.join(save_dir, f"{short}-head_heatmap_{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_causal_verification(result: CircuitResult, save_dir: str):
    """Bar chart: attribution vs causal effect for verified components."""
    if not result.verified_components:
        return None

    comps = sorted(result.verified_components, key=lambda c: c.causal_effect or 0)
    labels = []
    for c in comps:
        if c.component_type == "head":
            labels.append(f"L{c.layer}.H{c.head_idx}")
        else:
            labels.append(f"L{c.layer}.MLP")

    attributions = [c.attribution for c in comps]
    causal_effects = [c.causal_effect or 0 for c in comps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(comps) * 0.35)))

    y = np.arange(len(comps))

    # Attribution
    colors_attr = ['#4C72B0' if v >= 0 else '#C44E52' for v in attributions]
    ax1.barh(y, attributions, color=colors_attr, alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('Contrastive attribution')
    ax1.set_title('Attribution (projection onto refusal dir)')
    ax1.axvline(x=0, color='black', linewidth=0.5)

    # Causal effect
    colors_causal = ['#C44E52' if v < 0 else '#4C72B0' for v in causal_effects]
    ax2.barh(y, causal_effects, color=colors_causal, alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Causal effect (Δ refusal log-odds)')
    ax2.set_title('Causal verification (ablate component)')
    ax2.axvline(x=0, color='black', linewidth=0.5)

    plt.suptitle(f'Refusal Circuit — {result.model_name.split("/")[-1]}', fontsize=12, y=1.02)
    plt.tight_layout()

    short = result.model_name.split("/")[-1]
    path = os.path.join(save_dir, f"{short}-causal_verification.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_attribution_vs_causal(result: CircuitResult, save_dir: str):
    """Scatter plot: attribution (x) vs causal effect (y) per component."""
    if not result.verified_components:
        return None

    comps = result.verified_components
    attrs = [c.attribution for c in comps]
    causals = [c.causal_effect or 0 for c in comps]
    labels = []
    for c in comps:
        if c.component_type == "head":
            labels.append(f"L{c.layer}.H{c.head_idx}")
        else:
            labels.append(f"L{c.layer}.MLP")

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ['#DD8452' if c.component_type == 'mlp' else '#4C72B0' for c in comps]
    ax.scatter(attrs, causals, c=colors, s=60, alpha=0.8, edgecolors='black', linewidths=0.5)

    for i, label in enumerate(labels):
        ax.annotate(label, (attrs[i], causals[i]), fontsize=6, alpha=0.7,
                    xytext=(4, 4), textcoords='offset points')

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Contrastive attribution (projection)')
    ax.set_ylabel('Causal effect (Δ refusal log-odds)')
    ax.set_title(f'Attribution vs Causal Effect — {result.model_name.split("/")[-1]}')

    # Correlation
    if len(attrs) > 2:
        corr = np.corrcoef(attrs, causals)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    short = result.model_name.split("/")[-1]
    path = os.path.join(save_dir, f"{short}-attribution_vs_causal.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_cross_model_comparison(all_results: dict, save_dir: str):
    """Bar chart comparing top circuit components across models."""
    if len(all_results) < 2:
        return None

    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 6), sharey=False)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (model_name, result) in zip(axes, all_results.items()):
        if not result.verified_components:
            continue
        comps = sorted(result.verified_components, key=lambda c: c.causal_effect or 0)[:10]
        labels = []
        for c in comps:
            if c.component_type == "head":
                labels.append(f"L{c.layer}.H{c.head_idx}")
            else:
                labels.append(f"L{c.layer}.MLP")
        effects = [c.causal_effect or 0 for c in comps]
        colors = ['#C44E52' if v < 0 else '#4C72B0' for v in effects]

        y = np.arange(len(comps))
        ax.barh(y, effects, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(model_name.split("/")[-1], fontsize=10)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Causal Δ log-odds')

    plt.suptitle('Refusal Circuit Components Across Models', fontsize=12, y=1.02)
    plt.tight_layout()

    path = os.path.join(save_dir, "cross_model_circuit_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ── Serialization ───────────────────────────────────────────────────────────

def save_result(result: CircuitResult, save_dir: str):
    """Save CircuitResult as JSON for later analysis."""
    short = result.model_name.split("/")[-1]

    data = {
        "model_name": result.model_name,
        "direction_layer": result.direction_layer,
        "direction_pos": result.direction_pos,
        "num_layers": result.num_layers,
        "num_heads": result.num_heads,
        "contrastive_layer_attributions": [
            {"layer": a.layer_idx, "attn": a.attn_projection, "mlp": a.mlp_projection, "total": a.total_projection}
            for a in result.contrastive_layer_attributions
        ],
        "contrastive_head_attributions": [
            {"layer": a.layer_idx, "heads": a.head_projections, "mlp": a.mlp_projection}
            for a in result.contrastive_head_attributions
        ],
        "verified_components": [
            {
                "layer": c.layer, "type": c.component_type, "head_idx": c.head_idx,
                "attribution": c.attribution, "causal_effect": c.causal_effect,
            }
            for c in result.verified_components
        ],
    }

    path = os.path.join(save_dir, f"{short}-circuit.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def run_model(model_id: str, torch_dtype="auto") -> CircuitResult:
    """Run full circuit analysis on one model."""
    short = model_id.split("/")[-1]
    direction_path = f"results/{short}-refusal-direction"

    if not os.path.exists(f"{direction_path}.pt"):
        logger.warning(f"No saved direction for {short}, skipping. Run search first.")
        return None

    direction = DirectionVector.load(direction_path)
    logger.info(f"Loaded direction: layer={direction.layer}, pos={direction.position_index}, "
                f"score={direction.score:.4f}, norm={t.norm(direction.vector):.4f}")

    # Determine device
    if t.cuda.is_available():
        device = t.device("cuda:0")
    elif t.backends.mps.is_available():
        device = t.device("mps")
    else:
        device = t.device("cpu")

    logger.info(f"Loading {model_id} to {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # MPS float16 safety
    if device.type == "mps" and model.dtype == t.float16:
        logger.warning("Upcasting float16 -> bfloat16 on MPS")
        model = model.to(dtype=t.bfloat16)

    formatter = ChatPromptFormatter(tokenizer)
    applier = ModelInterventionApplier(model)
    analyzer = AttributionAnalyzer(model, tokenizer, applier.transformer_layers, formatter)

    # Get eval prompts
    eval_pos, eval_neg = create_refusal_eval_data()
    pos_prompts = eval_pos[:NUM_ATTRIBUTION_PROMPTS]
    neg_prompts = eval_neg[:NUM_ATTRIBUTION_PROMPTS]
    causal_prompts = eval_pos[:NUM_CAUSAL_PROMPTS]

    logger.info(f"Using {len(pos_prompts)} harmful + {len(neg_prompts)} benign for attribution, "
                f"{len(causal_prompts)} harmful for causal verification")

    # Run full circuit analysis
    start = time.time()
    result = analyzer.find_refusal_circuit(
        positive_prompts=pos_prompts,
        negative_prompts=neg_prompts,
        direction=direction,
        top_layers_k=TOP_LAYERS_K,
        top_components_k=TOP_COMPONENTS_K,
    )
    elapsed = time.time() - start
    logger.info(f"Circuit analysis for {short} completed in {elapsed:.1f}s")

    # Generate plots
    plot_layer_attributions(result, PLOTS_DIR)
    plot_head_heatmap(result, PLOTS_DIR, use_contrastive=True)
    plot_head_heatmap(result, PLOTS_DIR, use_contrastive=False)
    plot_causal_verification(result, PLOTS_DIR)
    plot_attribution_vs_causal(result, PLOTS_DIR)

    # Save JSON
    save_result(result, RESULTS_DIR)

    # Cleanup
    del model, tokenizer, analyzer, applier, formatter
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
    elif t.backends.mps.is_available():
        t.mps.empty_cache()

    return result


def print_cross_model_summary(all_results: dict):
    """Print a cross-model comparison table."""
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-MODEL REFUSAL CIRCUIT COMPARISON")
    logger.info("=" * 80)

    for model_name, result in all_results.items():
        short = model_name.split("/")[-1]
        logger.info(f"\n--- {short} (L{result.direction_layer}/pos{result.direction_pos}, "
                    f"{result.num_layers}L, {result.num_heads}H) ---")

        if not result.verified_components:
            logger.info("  No verified components")
            continue

        # Top 5 most causally important
        by_causal = sorted(result.verified_components, key=lambda c: c.causal_effect or 0)
        logger.info(f"  Top 5 by causal effect (most refusal reduction when ablated):")
        for c in by_causal[:5]:
            label = f"L{c.layer}.{'H' + str(c.head_idx) if c.component_type == 'head' else 'MLP'}"
            logger.info(f"    {label}: attr={c.attribution:+.4f}, causal={c.causal_effect:+.4f}")

        # Fraction of refusal from heads vs MLP
        head_causal = sum(c.causal_effect or 0 for c in result.verified_components if c.component_type == "head")
        mlp_causal = sum(c.causal_effect or 0 for c in result.verified_components if c.component_type == "mlp")
        total = head_causal + mlp_causal
        if abs(total) > 1e-6:
            logger.info(f"  Causal effect split: heads={head_causal:+.4f} ({100*head_causal/total:.0f}%), "
                        f"MLP={mlp_causal:+.4f} ({100*mlp_causal/total:.0f}%)")

        # Layer concentration
        causal_layers = set(c.layer for c in by_causal[:5] if (c.causal_effect or 0) < -0.1)
        if causal_layers:
            logger.info(f"  Circuit layers: {sorted(causal_layers)}")

    logger.info("\n" + "=" * 80)


def main():
    global TOP_LAYERS_K, TOP_COMPONENTS_K, NUM_ATTRIBUTION_PROMPTS, NUM_CAUSAL_PROMPTS

    parser = argparse.ArgumentParser(description="Refusal circuit analysis")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs to analyze (default: all)")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--top-layers", type=int, default=TOP_LAYERS_K)
    parser.add_argument("--top-components", type=int, default=TOP_COMPONENTS_K)
    parser.add_argument("--num-prompts", type=int, default=NUM_ATTRIBUTION_PROMPTS)
    args = parser.parse_args()

    TOP_LAYERS_K = args.top_layers
    TOP_COMPONENTS_K = args.top_components
    NUM_ATTRIBUTION_PROMPTS = args.num_prompts
    NUM_CAUSAL_PROMPTS = min(20, args.num_prompts)

    models = args.models or ALL_MODELS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_results = {}
    for model_id in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYZING: {model_id}")
        logger.info(f"{'='*60}")
        try:
            result = run_model(model_id, torch_dtype=args.torch_dtype)
            if result is not None:
                all_results[model_id] = result
        except Exception as e:
            logger.error(f"Failed on {model_id}: {e}", exc_info=True)
            continue

    if len(all_results) >= 2:
        plot_cross_model_comparison(all_results, PLOTS_DIR)

    print_cross_model_summary(all_results)


if __name__ == "__main__":
    main()
