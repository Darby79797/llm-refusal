import os
import numpy as np
import torch as t
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from tqdm import tqdm
import logging

from datatypes import PromptData, DirectionVector, DirectionScores
from concept import DEFAULT_SEARCH_CONFIG
from direction_methods import DirectionMethod
from scoring import Three_Score_Evaluator
from interventions import ModelInterventionApplier

logger = logging.getLogger(__name__)


class DirectionFinder:
    """Finds the best direction vector based on multi-objective criteria."""
    def __init__(self, model, tokenizer, intervention_applier, prompt_formatter,
                 direction_method: DirectionMethod,
                 evaluator: Three_Score_Evaluator,
                 search_config: Optional[dict] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.intervention_applier = intervention_applier
        self.prompt_formatter = prompt_formatter
        self.direction_finder_method = direction_method
        self.evaluator = evaluator
        self.search_config = search_config or DEFAULT_SEARCH_CONFIG

    def _print_debug_info(self, name: str, info: Dict):
        """Helper function to print debug information for the best candidates."""
        if info['dir']:
            s = info['scores']
            d = info['dir']
            logger.info(f"  Best {name:<7}: Layer {d.layer:2d}, Pos {d.position_index:2d} | Bypass: {s.bypass:7.4f}, Induce: {s.induce:7.4f}, Induce(global): {s.induce_global:7.4f}, KL: {s.kl:7.4f}")
        else:
            logger.info(f"  No candidate found for Best {name}")

    def _plot_and_save_search_results(self, results_df: pd.DataFrame, model_short_name: str):
        """
        Plots and saves the search results for induce and bypass scores against layer,
        with different lines for each token position.
        """
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

        # --- Induce Score Plot ---
        plt.figure(figsize=(14, 8))
        try:
            pivot_induce = results_df.pivot(index='layer', columns='position', values='induce_score')
            for pos in sorted(pivot_induce.columns):
                plt.plot(pivot_induce.index, pivot_induce[pos], marker='o', linestyle='-', label=f'Position {pos}')
            plt.title(f'Induce Score vs. Layer for {model_short_name}', fontsize=16)
            plt.xlabel('Layer Index', fontsize=12)
            plt.ylabel('Induce Score (Higher is Better)', fontsize=12)
            plt.legend(title='Token Position')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            induce_plot_path = os.path.join(plot_dir, f"{model_short_name}-induce_score_vs_layer.png")
            plt.savefig(induce_plot_path)
            logger.info(f"Saved induce score plot to {induce_plot_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate or save induce score plot: {e}")

        # --- Bypass Score Plot ---
        plt.figure(figsize=(14, 8))
        try:
            pivot_bypass = results_df.pivot(index='layer', columns='position', values='bypass_score')
            for pos in sorted(pivot_bypass.columns):
                plt.plot(pivot_bypass.index, pivot_bypass[pos], marker='o', linestyle='-', label=f'Position {pos}')
            plt.title(f'Bypass Score vs. Layer for {model_short_name}', fontsize=16)
            plt.xlabel('Layer Index', fontsize=12)
            plt.ylabel('Bypass Score (Lower is Better)', fontsize=12)
            plt.legend(title='Token Position')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            bypass_plot_path = os.path.join(plot_dir, f"{model_short_name}-bypass_score_vs_layer.png")
            plt.savefig(bypass_plot_path)
            logger.info(f"Saved bypass score plot to {bypass_plot_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate or save bypass score plot: {e}")

    def find_best_direction(
        self, train_data: PromptData, val_data: PromptData, max_positions: Optional[int] = None,
        output_prefix: Optional[str] = None,
    ) -> Optional[DirectionVector]:
        """
        Selects a direction vector based on strict multi-objective criteria.
        Also logs a summary of the best candidates found for each metric.

        output_prefix names the saved plots and the per-candidate score CSV
        (e.g. "<model>-<concept>"); defaults to the model name.
        """
        if max_positions is None:
            cfg_positions = self.search_config.get("max_positions", "auto")
            if cfg_positions == "auto":
                apt = getattr(self.prompt_formatter, 'assistant_prefix_tokens', 0)
                max_positions = apt + 1
                logger.info(f"Auto max_positions: assistant_prefix_tokens={apt} → "
                            f"max_positions={max_positions} (covering pos -1 through -{max_positions}, EOI ≈ -{apt + 1})")
            else:
                max_positions = cfg_positions
        logger.info(f"Computing difference-in-means vectors (max_positions={max_positions})...")
        difference_vectors = self.direction_finder_method.compute_difference_vectors(train_data, max_positions)

        logger.info("Pre-computing baseline scores and logits on the validation set...")
        positive_prompts = val_data.positive
        negative_prompts = val_data.negative

        baseline_pos_logits = self.evaluator._get_logits(positive_prompts) if positive_prompts else []
        baseline_neg_logits = self.evaluator._get_logits(negative_prompts) if negative_prompts else []

        baseline_bypass_score = np.nanmean([self.evaluator.metric.compute_log_odds(logits) for logits in baseline_pos_logits]) if baseline_pos_logits else 0.0
        baseline_induce_score = np.nanmean([self.evaluator.metric.compute_log_odds(logits) for logits in baseline_neg_logits]) if baseline_neg_logits else 0.0

        logger.info(f"Baseline Scores | Bypass: {baseline_bypass_score:7.4f}, Induce: {baseline_induce_score:7.4f}, KL: 0.0")

        num_layers = len(self.intervention_applier.transformer_layers)
        layer_cutoff = int(self.search_config["layer_cutoff_frac"] * num_layers)

        logger.info(f"Evaluating direction candidates with multi-objective criteria...")

        all_scores_data = []
        all_candidates = []  # (direction, scores) for fallback selection
        best_bypass_info = {'score': float('inf'), 'dir': None, 'scores': None}
        best_induce_info = {'score': float('-inf'), 'dir': None, 'scores': None}
        best_kl_info = {'score': float('inf'), 'dir': None, 'scores': None}

        candidate_iterator = tqdm(difference_vectors.items(), desc="Evaluating candidates")
        for (layer, pos_idx), vec in candidate_iterator:
            # Layer 0 is the raw token embeddings: no context has been mixed in, so a
            # difference in means there is a token-identity contrast, not a behavior
            # (and at the post-instruction template positions it is exactly 0).
            if layer == 0 or layer >= layer_cutoff:
                continue

            current_direction = DirectionVector(vector=vec, layer=layer, position_index=pos_idx, score=0)
            scores = self.evaluator.compute_all_scores(current_direction, val_data, baseline_neg_logits=baseline_neg_logits)

            all_scores_data.append({
                'layer': layer,
                'position': pos_idx,
                'bypass_score': scores.bypass,
                'induce_score': scores.induce,
                'induce_global_score': scores.induce_global,
                'kl_score': scores.kl,
            })
            # Non-finite scores come from (near-)zero difference vectors, whose unit
            # vector is NaN. Keep them out of selection (NaN also breaks the
            # fallback's sort order).
            if not all(np.isfinite([scores.bypass, scores.induce, scores.induce_global, scores.kl])):
                logger.debug(f"Skipping candidate L{layer} P{pos_idx}: non-finite scores {scores}")
                continue
            all_candidates.append((current_direction, scores))

            if scores.bypass < best_bypass_info['score']:
                best_bypass_info.update({'score': scores.bypass, 'dir': current_direction, 'scores': scores})
            if scores.induce > best_induce_info['score']:
                best_induce_info.update({'score': scores.induce, 'dir': current_direction, 'scores': scores})
            if scores.kl < best_kl_info['score']:
                best_kl_info.update({'score': scores.kl, 'dir': current_direction, 'scores': scores})


        if all_scores_data:
            results_df = pd.DataFrame(all_scores_data)
            if output_prefix is None:
                output_prefix = self.model.name_or_path.split('/')[-1] if hasattr(self.model, 'name_or_path') else 'unknown_model'
            os.makedirs("results", exist_ok=True)
            csv_path = os.path.join("results", f"{output_prefix}-search-scores.csv")
            results_df.to_csv(csv_path, index=False)
            logger.info(f"Saved per-candidate search scores to {csv_path}")
            self._plot_and_save_search_results(results_df, output_prefix)
        else:
            logger.warning("No data was collected during search; skipping data saving and plotting.")

        selected_direction = self._select_strict(all_candidates)
        if selected_direction:
            logger.info(f"\n--- Strictly Selected Direction (Met All Criteria) ---")
            logger.info(f"Layer: {selected_direction.layer}, Position: {selected_direction.position_index}")
            logger.info(f"Final Bypass Score: {selected_direction.score:.4f}")
        else:
            logger.warning("\nNo direction met strict criteria (induce > %.2f, KL < %.2f). Trying progressive relaxation...",
                           self.search_config["induce_threshold"], self.search_config["kl_threshold"])
            logger.info(f"  {'Baseline':<7}:               | Bypass: {baseline_bypass_score:7.4f}, Induce: {baseline_induce_score:7.4f}, KL: {0.0:7.4f}")
            self._print_debug_info("Bypass", best_bypass_info)
            self._print_debug_info("Induce", best_induce_info)
            self._print_debug_info("KL", best_kl_info)

            induce_mode = self.search_config.get("induce_mode", "single_layer")
            selected_direction = self._progressive_fallback(
                all_candidates, baseline_induce_score, induce_mode=induce_mode
            )

        return selected_direction

    def _select_strict(self, candidates: list) -> Optional[DirectionVector]:
        """
        Strict selection: among candidates with induce > induce_threshold and
        KL < kl_threshold, take the lowest bypass score (Arditi's rule).

        bypass_tie_frac > 0 treats every passing candidate whose bypass is within
        that fraction of the best as tied, and breaks the tie by the highest
        induce score. Bypass differences that small are within val-set noise,
        and without the tie-break the rule can pick a near-tie with a much
        weaker induce score (Llama-3.1 refusal: L12/P-2, bypass -12.72, induce
        1.08, gives 85% induction; L11/P-1, bypass -12.39, induce 7.15, gives 100%).
        """
        induce_mode = self.search_config.get("induce_mode", "single_layer")
        tie_frac = self.search_config.get("bypass_tie_frac", 0.0)

        def _get_induce(scores):
            return scores.induce_global if induce_mode == "all_layers" else scores.induce

        passing = [
            (d, s) for d, s in candidates
            if _get_induce(s) > self.search_config["induce_threshold"]
            and s.kl < self.search_config["kl_threshold"]
        ]
        if not passing:
            return None
        best_bypass = min(s.bypass for _, s in passing)
        tied = [(d, s) for d, s in passing if s.bypass <= best_bypass + tie_frac * abs(best_bypass)]
        best_dir, best_scores = max(tied, key=lambda x: (_get_induce(x[1]), -x[1].bypass))
        if len(tied) > 1:
            logger.info(f"  {len(tied)} candidates within {tie_frac:.0%} of the best bypass "
                        f"({best_bypass:.4f}); taking the highest induce")
        best_dir.score = best_scores.bypass
        return best_dir

    def _progressive_fallback(
        self,
        candidates: list,
        baseline_induce: float,
        induce_mode: str = "single_layer",
    ) -> Optional[DirectionVector]:
        """
        Progressive relaxation fallback: filter by increasingly relaxed induce
        delta thresholds, rank by induce (highest first) among passing candidates.

        On small models, bypass and induce rankings diverge — layers with the best
        ablation properties often have poor induction. Since induction is the
        better predictor of actual generation behavior, we rank by induce within
        each tier rather than bypass.

        induce_mode: "single_layer" uses scores.induce, "all_layers" uses scores.induce_global.
        """
        def _get_induce(scores):
            return scores.induce_global if induce_mode == "all_layers" else scores.induce

        # Relaxation tiers: (min induce delta above baseline, max KL)
        # Calibrated against behavioral induction rates on Qwen2.5-{0.5B,1.5B,3B,7B}.
        # Good layers consistently have KL > 1.0, so KL thresholds are generous.
        # The layer_cutoff_frac (0.65) does the heavy lifting to exclude deep layers
        # where high induce scores don't translate to actual behavioral induction.
        tiers = [
            (3.0, 5.0,  "induce Δ>+3, KL<5.0"),
            (1.5, 10.0, "induce Δ>+1.5, KL<10.0"),
            (0.0, 20.0, "induce Δ>+0, KL<20.0"),
        ]

        mode_label = f" [{induce_mode}]" if induce_mode != "single_layer" else ""

        for min_delta, max_kl, label in tiers:
            passing = [
                (d, s) for d, s in candidates
                if (_get_induce(s) - baseline_induce) > min_delta and s.kl < max_kl
            ]
            if passing:
                # Rank by induce (highest = best induction) among passing candidates
                passing.sort(key=lambda x: _get_induce(x[1]), reverse=True)
                best_dir, best_scores = passing[0]
                induce_val = _get_induce(best_scores)
                best_dir.score = induce_val
                delta = induce_val - baseline_induce
                logger.info(f"  Relaxed selection ({label}){mode_label}: "
                            f"Layer {best_dir.layer}, Pos {best_dir.position_index} | "
                            f"Bypass: {best_scores.bypass:.4f}, Induce: {best_scores.induce:.4f}, "
                            f"Induce(global): {best_scores.induce_global:.4f} "
                            f"(Δ={delta:+.4f}), KL: {best_scores.kl:.4f}")
                return best_dir

        # Absolute last resort: best induce candidate regardless of other metrics
        if candidates:
            candidates.sort(key=lambda x: _get_induce(x[1]), reverse=True)
            best_dir, best_scores = candidates[0]
            best_dir.score = _get_induce(best_scores)
            logger.warning(f"  No candidates passed any relaxed tier. "
                           f"Falling back to best induce{mode_label}: Layer {best_dir.layer}, Pos {best_dir.position_index}")
            return best_dir

        return None
