"""Cross-concept analysis: geometric and behavioral relationships between direction vectors."""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import json

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
from coherence import is_degenerate

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


# ── Behavioral analysis ─────────────────────────────────────────

class CellRunner:
    """Runs and memoizes cross-concept cells: (ablated directions) x (measured concept).

    Every cell generates over the measured concept's full positive eval set with one
    batch size shared by all cells (bf16 numerics depend on batch shape, so cells are
    only comparable if they share it), and records the continuous log-odds metric and
    a degeneracy flag beside the detection rate: ablating a direction can break the
    model, and a broken model reads as "0% detected" under any phrase heuristic.
    Memoization matters because the multi-ablation test reuses the baseline and
    single-ablation cells of the interference matrix.
    """

    def __init__(self, model, tokenizer, intervention_applier, prompt_formatter,
                 directions: List[DirectionVector], concepts: List[ConceptDefinition],
                 max_prompts: Optional[int] = None, gen_batch_size: Union[int, str] = "auto",
                 max_new_tokens: int = 64):
        self.tokenizer = tokenizer
        self.intervention_applier = intervention_applier
        self.directions = {c.name: d for c, d in zip(concepts, directions)}
        self.max_new_tokens = max_new_tokens
        proxy = _FrameworkProxy(model, tokenizer, intervention_applier, prompt_formatter)
        self.evaluators, self.prompts = {}, {}
        for concept in concepts:
            self.evaluators[concept.name] = BigEvaluator(
                proxy, detection_phrases=concept.detection_phrases,
                detection_fn=concept.detection_fn, judge_prompt=concept.judge_prompt,
                gen_batch_size=gen_batch_size, target_tokens=concept.target_tokens,
            )
            pos_prompts, _ = concept.eval_data_fn()
            self.prompts[concept.name] = pos_prompts[:max_prompts]
        all_prompts = [p for ps in self.prompts.values() for p in ps]
        first = next(iter(self.evaluators.values()))
        self.batch_size = first.resolve_batch_size(all_prompts, max_new_tokens)
        logger.info(f"Cross-concept config: gen_batch_size={self.batch_size} (requested {gen_batch_size}), "
                    f"max_new_tokens={max_new_tokens}, prompts per concept="
                    f"{ {k: len(v) for k, v in self.prompts.items()} }")
        self.cells: Dict[str, Dict] = {}
        self.generations: Dict[str, List[Dict]] = {}

    @staticmethod
    def key(ablated: List[str], measured: str) -> str:
        return f"ablate[{'+'.join(sorted(ablated))}]->{measured}" if ablated else f"baseline->{measured}"

    def run(self, ablated: List[str], measured: str) -> Dict:
        """Detection rate, log-odds and degenerate rate for `measured` with the span
        of `ablated` removed at every layer (joint ablation if more than one)."""
        key = self.key(ablated, measured)
        if key in self.cells:
            return self.cells[key]
        if ablated:
            # One [k, d] stack: the span is removed in a single order-independent
            # projection (k=1 is plain single-direction ablation).
            stack = t.stack([self.directions[n].vector for n in ablated])
            with self.intervention_applier.intervened(
                    stack, "ablate", 1.0,
                    layers=list(range(len(self.intervention_applier.transformer_layers)))):
                cell = self._measure(measured, key)
        else:
            cell = self._measure(measured, key)
        self.cells[key] = cell
        logger.info(f"  {key}: rate={cell['rate']:.3f} log_odds={cell['log_odds']} "
                    f"degenerate={cell['degenerate_rate']:.3f}")
        return cell

    def _measure(self, measured: str, key: str) -> Dict:
        evaluator, prompts = self.evaluators[measured], self.prompts[measured]
        texts = evaluator.generate_responses(prompts, batch_size=self.batch_size,
                                             max_new_tokens=self.max_new_tokens)
        entries = [
            {"prompt": p, "response": r, "detected": evaluator._check_for_detection(r),
             "degenerate": is_degenerate(self.tokenizer.encode(r, add_special_tokens=False))}
            for p, r in zip(prompts, texts)
        ]
        self.generations[key] = entries
        return {
            "rate": evaluator.evaluate_detection_rate(prompts, generated_texts=texts),
            "log_odds": evaluator._log_odds_metric(prompts, batch_size=self.batch_size),
            "degenerate_rate": sum(e["degenerate"] for e in entries) / len(entries) if entries else 0.0,
            "n": len(entries),
        }

    def oom_splits(self) -> int:
        return sum(e.oom_splits for e in self.evaluators.values())


def measure_interference(runner: CellRunner, names: List[str]) -> np.ndarray:
    """Interference matrix: entry (i,j) = change in concept j's detection rate when concept i is ablated.

    Diagonal = concept's own ablation effect (sanity check).
    """
    n = len(names)
    baselines = [runner.run([], name)["rate"] for name in names]
    interference = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            interference[i, j] = runner.run([names[i]], names[j])["rate"] - baselines[j]
    return interference


def run_multi_ablation(runner: CellRunner, ablate_names: List[str], measure_name: str) -> Dict[str, float]:
    """Ablate multiple directions simultaneously, measure one concept's detection rate.

    Returns dict with 'individual_sum', 'simultaneous', and per-concept individual rates.
    The simultaneous condition jointly ablates the span of all named directions in one
    order-independent projection (interventions.py's [k, d] ablate), rather than stacking
    single-direction ablations, which would apply sequential (order-dependent)
    projection removal for non-orthogonal directions.
    """
    baseline_rate = runner.run([], measure_name)["rate"]
    results = {"baseline": baseline_rate}
    individual_deltas = {}
    for name in ablate_names:
        rate = runner.run([name], measure_name)["rate"]
        individual_deltas[name] = rate - baseline_rate
        results[f"individual_{name}"] = rate
    results["individual_sum"] = sum(individual_deltas.values())
    simultaneous_rate = runner.run(list(ablate_names), measure_name)["rate"]
    results["simultaneous"] = simultaneous_rate
    results["simultaneous_delta"] = simultaneous_rate - baseline_rate
    logger.info(f"Multi-ablation on {measure_name}: {results}")
    return results


def _save_cells(runner: CellRunner, names, sim_matrix, pca_var, interference, multi_results, path: str):
    """Every cell's metrics and every response, so a rate can be audited against the text."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "concepts": names,
            "directions": {n: {"layer": d.layer, "position_index": d.position_index}
                           for n, d in runner.directions.items()},
            "gen_batch_size": runner.batch_size,
            "oom_splits": runner.oom_splits(),
            "similarity_matrix": np.asarray(sim_matrix).tolist(),
            "pca_explained_variance": np.asarray(pca_var).tolist(),
            "interference_matrix": np.asarray(interference).tolist(),
            "multi_ablation": multi_results,
            "cells": runner.cells,
            "generations": runner.generations,
        }, f, indent=1)
    logger.info(f"Saved cross-concept cells and generations to {path}")


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
    gen_batch_size: Union[int, str] = "auto",
    max_prompts: Optional[int] = None,
    output_path: Optional[str] = None,
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
        runner = CellRunner(model, tokenizer, intervention_applier, prompt_formatter,
                             directions, concepts, max_prompts=max_prompts,
                             gen_batch_size=gen_batch_size)
        interference = measure_interference(runner, names)
        plot_interference_matrix(names, interference, model_name)

        # 4. Multi-ablation composition test (if 3+ concepts; with 2, "the others"
        # is a single direction and the cell is already in the interference matrix)
        multi_results = {}
        if len(concepts) >= 3:
            for name in names:
                others = [n for n in names if n != name]
                multi_results[name] = run_multi_ablation(runner, ablate_names=others, measure_name=name)
        if output_path:
            _save_cells(runner, names, sim_matrix, pca_var, interference, multi_results, output_path)
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
        for key, cell in runner.cells.items():
            logger.info(f"  {key}: rate={cell['rate']:.3f} log_odds={cell['log_odds']} "
                        f"degenerate={cell['degenerate_rate']:.3f} (n={cell['n']})")
        logger.info(f"  gen_batch_size={runner.batch_size}"
                    + (f", WARNING: {runner.oom_splits()} batch(es) split on OOM" if runner.oom_splits() else ""))
    logger.info("=" * 60)

    return result
