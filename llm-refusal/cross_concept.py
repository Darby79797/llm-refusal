"""Cross-concept analysis: geometric and behavioral relationships between direction vectors."""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import torch as t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datatypes import DirectionVector
from concept import ConceptDefinition, get_concept
from interventions import ModelInterventionApplier
from formatting import ChatPromptFormatter
from evaluation import BigEvaluator

logger = logging.getLogger(__name__)


@dataclass
class CrossConceptResult:
    concept_names: List[str]
    similarity_matrix: np.ndarray            # (n, n) cosine similarities
    interference_matrix: Optional[np.ndarray] = None  # (n, n) detection rate deltas
    pca_explained_variance: Optional[np.ndarray] = None
    multi_ablation_results: Dict[str, float] = field(default_factory=dict)


class _FrameworkProxy:
    """Minimal proxy that satisfies BigEvaluator's framework interface."""
    def __init__(self, model, tokenizer, intervention_applier, prompt_formatter):
        self.model = model
        self.tokenizer = tokenizer
        self.intervention_applier = intervention_applier
        self.prompt_formatter = prompt_formatter


# ── Geometric analysis ──────────────────────────────────────────

def compute_pairwise_cosine_similarity(directions: List[DirectionVector]) -> np.ndarray:
    """Cosine similarity matrix between unit direction vectors."""
    n = len(directions)
    units = [d.unit.float() for d in directions]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = t.dot(units[i], units[j]).item()
    return sim


def analyze_direction_subspace(directions: List[DirectionVector]) -> np.ndarray:
    """Uncentered SVD on stacked unit vectors — returns explained variance ratios.

    Direction vectors are meaningful relative to the origin (they're rays in activation
    space, not a point cloud), so mean-centering — which sklearn's PCA does — measures
    variance around the centroid instead of the subspace actually spanned by the vectors.
    Centering also produces NaN explained_variance_ratio for near-identical inputs (two
    identical unit vectors center to all-zero rows, giving 0/0). Uncentered SVD avoids both:
    singular values of the stacked (unit-vector) matrix directly give the spanned subspace's
    variance decomposition, and duplicate/near-duplicate vectors just collapse rank cleanly.
    """
    units = np.stack([d.unit.float().numpy() for d in directions])
    n_components = min(len(directions), units.shape[1])
    _, s, _ = np.linalg.svd(units, full_matrices=False)
    s = s[:n_components]
    total = float(np.sum(s ** 2))
    if total <= 0.0:
        return np.zeros(n_components)
    return (s ** 2) / total


# ── Joint (order-independent) multi-direction ablation ──────────

def _joint_ablation_basis(directions: List[DirectionVector], rtol: float = 1e-5) -> t.Tensor:
    """Orthonormal basis (d, r) spanning the union of `directions`' unit vectors.

    Order-independent by construction: depends only on the subspace spanned by the
    vectors (via SVD of the stacked unit vectors), not on the order they're passed in.
    Rank-deficient input (e.g. duplicate or parallel directions) collapses to a lower-
    rank basis via singular-value thresholding, so ablating [A, A] reduces to ablating
    [A] alone rather than double-counting the same direction.
    """
    stacked = t.stack([d.unit.to(dtype=t.float64) for d in directions], dim=0)  # (k, d)
    _, s, vh = t.linalg.svd(stacked, full_matrices=False)  # vh: (min(k,d), d)
    threshold = s.max().item() * rtol if s.numel() > 0 else 0.0
    keep = s > threshold
    basis = vh[keep]  # (r, d) orthonormal rows spanning the subspace
    return basis.T.contiguous()  # (d, r)


def apply_joint_ablation(
    intervention_applier: ModelInterventionApplier,
    directions: List[DirectionVector],
    layers: Optional[List[int]] = None,
) -> list:
    """Registers hooks that jointly ablate the subspace spanned by `directions`.

    `ModelInterventionApplier.apply_direction_intervention` only ablates one direction at
    a time; stacking calls for multiple directions applies sequential projection removal,
    which is order-dependent whenever the directions are non-orthogonal (removing A then B
    is not the same operation as removing B then A). This instead computes an orthonormal
    basis Q for span(directions) once and removes the whole subspace in a single joint
    projection: x - (x @ Q) @ Q.T. That is order-independent and correctly handles
    linearly-dependent directions (see `_joint_ablation_basis`).

    Mirrors the 3-hooks-per-layer pattern used by `apply_direction_intervention` for
    single-direction ablation (block pre-hook + self_attn post-hook + mlp post-hook),
    since the base API has no way to express a multi-direction joint ablation.

    Returns the list of hook handles; the caller is responsible for removing them
    (e.g. via `clear_joint_ablation_hooks` in a `finally` block).
    """
    if layers is None:
        layers = list(range(len(intervention_applier.transformer_layers)))
    model_dtype = intervention_applier.model.dtype
    basis = _joint_ablation_basis(directions).to(dtype=model_dtype)  # (d, r)

    def remove_span(hidden_states, basis):
        q = basis.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states - t.matmul(t.matmul(hidden_states, q), q.T)

    def make_block_pre_hook(basis):
        def hook(module, args):
            hidden_states = args[0]
            modified_states = remove_span(hidden_states, basis)
            return (modified_states,) + args[1:]
        return hook

    def make_sublayer_post_hook(basis):
        def hook(module, input, output):
            is_tuple_output = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple_output else output
            modified_states = remove_span(hidden_states, basis)
            if is_tuple_output:
                return (modified_states,) + output[1:]
            else:
                return modified_states
        return hook

    block_hook_fn = make_block_pre_hook(basis)
    sublayer_hook_fn = make_sublayer_post_hook(basis)

    hooks = []
    for layer_idx in layers:
        if 0 <= layer_idx < len(intervention_applier.transformer_layers):
            block = intervention_applier.transformer_layers[layer_idx]
            hooks.append(block.register_forward_pre_hook(block_hook_fn))
            attn, mlp = intervention_applier._get_sublayers(block)
            hooks.append(attn.register_forward_hook(sublayer_hook_fn))
            hooks.append(mlp.register_forward_hook(sublayer_hook_fn))
    return hooks


def clear_joint_ablation_hooks(hooks: list) -> None:
    """Removes hooks returned by `apply_joint_ablation`."""
    for hook in hooks:
        hook.remove()


# ── Behavioral analysis ─────────────────────────────────────────

def measure_interference(
    model,
    tokenizer,
    intervention_applier: ModelInterventionApplier,
    prompt_formatter: ChatPromptFormatter,
    directions: List[DirectionVector],
    concepts: List[ConceptDefinition],
    max_prompts: int = 30,
) -> np.ndarray:
    """Interference matrix: entry (i,j) = change in concept j's detection rate when concept i is ablated.

    Diagonal = concept's own ablation effect (sanity check).
    """
    n = len(directions)
    num_layers = len(intervention_applier.transformer_layers)
    proxy = _FrameworkProxy(model, tokenizer, intervention_applier, prompt_formatter)

    # Precompute baseline detection rates for each concept
    baselines = []
    for concept in concepts:
        evaluator = BigEvaluator(
            proxy, detection_phrases=concept.detection_phrases,
            detection_fn=concept.detection_fn, judge_prompt=concept.judge_prompt,
        )
        pos_prompts, _ = concept.eval_data_fn()
        pos_prompts = pos_prompts[:max_prompts]
        baseline_rate = evaluator.evaluate_detection_rate(pos_prompts)
        baselines.append((evaluator, pos_prompts, baseline_rate))
        logger.info(f"  Baseline detection rate for {concept.name}: {baseline_rate:.3f}")

    interference = np.zeros((n, n))
    for i in range(n):
        # Ablate direction i globally
        intervention_applier.apply_direction_intervention(
            directions[i], "ablate", 1.0, layers=list(range(num_layers))
        )
        try:
            for j in range(n):
                evaluator_j, prompts_j, baseline_j = baselines[j]
                rate_with_ablation = evaluator_j.evaluate_detection_rate(prompts_j)
                interference[i, j] = rate_with_ablation - baseline_j
                logger.info(
                    f"  Ablate {concepts[i].name} → {concepts[j].name} detection: "
                    f"{baseline_j:.3f} → {rate_with_ablation:.3f} (delta={interference[i, j]:+.3f})"
                )
        finally:
            # Ensure hooks are removed even if evaluation raises, so a failure on one
            # direction doesn't leak hooks into subsequent iterations or callers.
            intervention_applier.clear_interventions()

    return interference


def test_multi_ablation(
    model,
    tokenizer,
    intervention_applier: ModelInterventionApplier,
    prompt_formatter: ChatPromptFormatter,
    directions: List[DirectionVector],
    concepts: List[ConceptDefinition],
    ablate_names: List[str],
    measure_name: str,
    max_prompts: int = 30,
) -> Dict[str, float]:
    """Ablate multiple directions simultaneously, measure one concept's detection rate.

    Returns dict with 'individual_sum', 'simultaneous', and per-concept individual deltas.
    """
    num_layers = len(intervention_applier.transformer_layers)
    proxy = _FrameworkProxy(model, tokenizer, intervention_applier, prompt_formatter)

    # Find the concept to measure
    measure_idx = next(i for i, c in enumerate(concepts) if c.name == measure_name)
    measure_concept = concepts[measure_idx]
    evaluator = BigEvaluator(
        proxy, detection_phrases=measure_concept.detection_phrases,
        detection_fn=measure_concept.detection_fn, judge_prompt=measure_concept.judge_prompt,
    )
    pos_prompts, _ = measure_concept.eval_data_fn()
    pos_prompts = pos_prompts[:max_prompts]

    # Baseline
    baseline_rate = evaluator.evaluate_detection_rate(pos_prompts)
    results = {"baseline": baseline_rate}

    # Individual ablations
    individual_deltas = {}
    for name in ablate_names:
        idx = next(i for i, c in enumerate(concepts) if c.name == name)
        intervention_applier.apply_direction_intervention(
            directions[idx], "ablate", 1.0, layers=list(range(num_layers))
        )
        try:
            rate = evaluator.evaluate_detection_rate(pos_prompts)
            individual_deltas[name] = rate - baseline_rate
            results[f"individual_{name}"] = rate
        finally:
            intervention_applier.clear_interventions()

    results["individual_sum"] = sum(individual_deltas.values())

    # Simultaneous ablation — jointly ablate the span of all named directions in one
    # order-independent projection (see apply_joint_ablation), rather than stacking
    # single-direction ablations, which would apply sequential (order-dependent)
    # projection removal for non-orthogonal directions.
    idxs = [next(i for i, c in enumerate(concepts) if c.name == name) for name in ablate_names]
    joint_directions = [directions[idx] for idx in idxs]
    hooks = apply_joint_ablation(intervention_applier, joint_directions, layers=list(range(num_layers)))
    try:
        simultaneous_rate = evaluator.evaluate_detection_rate(pos_prompts)
    finally:
        clear_joint_ablation_hooks(hooks)
    results["simultaneous"] = simultaneous_rate
    results["simultaneous_delta"] = simultaneous_rate - baseline_rate

    logger.info(f"Multi-ablation on {measure_name}: {results}")
    return results


# ── Plotting ─────────────────────────────────────────────────────

def plot_similarity_heatmap(names: List[str], sim_matrix: np.ndarray, model_name: str) -> str:
    """Save cosine similarity heatmap and return path."""
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im)
    ax.set_title(f"Direction Cosine Similarity — {model_name}")
    plt.tight_layout()
    path = f"plots/{model_name.split('/')[-1]}-cross_concept_similarity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved similarity heatmap: {path}")
    return path


def plot_interference_matrix(names: List[str], interference: np.ndarray, model_name: str) -> str:
    """Save interference matrix heatmap and return path."""
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    vabs = max(abs(interference.min()), abs(interference.max()), 0.01)
    im = ax.imshow(interference, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n(measured)" for n in names], rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}\n(ablated)" for n in names])
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{interference[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im)
    ax.set_title(f"Intervention Interference — {model_name}")
    plt.tight_layout()
    path = f"plots/{model_name.split('/')[-1]}-cross_concept_interference.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved interference heatmap: {path}")
    return path


# ── Orchestrator ─────────────────────────────────────────────────

def run_cross_concept_analysis(
    model,
    tokenizer,
    intervention_applier: ModelInterventionApplier,
    prompt_formatter: ChatPromptFormatter,
    directions: List[DirectionVector],
    concepts: List[ConceptDefinition],
    model_name: str,
    run_interference: bool = True,
) -> CrossConceptResult:
    """Top-level orchestrator for cross-concept analysis."""
    names = [c.name for c in concepts]
    logger.info(f"=== Cross-Concept Analysis: {names} ===")

    # 1. Cosine similarity
    logger.info("Computing pairwise cosine similarities...")
    sim_matrix = compute_pairwise_cosine_similarity(directions)
    logger.info(f"Similarity matrix:\n{sim_matrix}")
    plot_similarity_heatmap(names, sim_matrix, model_name)

    # 2. PCA
    logger.info("Analyzing direction subspace (PCA)...")
    pca_var = analyze_direction_subspace(directions)
    logger.info(f"PCA explained variance ratios: {pca_var}")

    # 3. Interference (optional, expensive)
    interference = None
    if run_interference:
        logger.info("Measuring intervention interference...")
        interference = measure_interference(
            model, tokenizer, intervention_applier, prompt_formatter,
            directions, concepts,
        )
        plot_interference_matrix(names, interference, model_name)

        # 4. Multi-ablation composition test (if 2+ concepts)
        multi_results = {}
        if len(concepts) >= 2:
            for concept in concepts:
                others = [c.name for c in concepts if c.name != concept.name]
                result = test_multi_ablation(
                    model, tokenizer, intervention_applier, prompt_formatter,
                    directions, concepts,
                    ablate_names=others, measure_name=concept.name,
                )
                multi_results[concept.name] = result
    else:
        multi_results = {}

    result = CrossConceptResult(
        concept_names=names,
        similarity_matrix=sim_matrix,
        interference_matrix=interference,
        pca_explained_variance=pca_var,
        multi_ablation_results=multi_results,
    )

    # Summary log
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-CONCEPT ANALYSIS SUMMARY")
    logger.info("=" * 60)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            logger.info(f"  cos({names[i]}, {names[j]}) = {sim_matrix[i, j]:.4f}")
    logger.info(f"  PCA explained variance: {pca_var}")
    if interference is not None:
        logger.info(f"  Interference matrix diagonal (self-ablation effect): "
                     f"{[f'{names[i]}: {interference[i, i]:+.3f}' for i in range(len(names))]}")
    logger.info("=" * 60)

    return result
