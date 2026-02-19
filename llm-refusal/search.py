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
        self.direction_method = direction_method
        self.direction_finder_method = self.direction_method  # backward compat alias
        self.evaluator = evaluator
        self.search_config = search_config or DEFAULT_SEARCH_CONFIG

    def _print_debug_info(self, name: str, info: Dict):
        """Helper function to print debug information for the best candidates."""
        if info['dir']:
            s = info['scores']
            d = info['dir']
            logger.info(f"  Best {name:<7}: Layer {d.layer:2d}, Pos {d.position_index:2d} | Bypass: {s.bypass:7.4f}, Induce: {s.induce:7.4f}, KL: {s.kl:7.4f}")
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
        self, train_data: PromptData, val_data: PromptData, max_positions: int = 5
    ) -> Optional[DirectionVector]:
        """
        Selects a direction vector based on strict multi-objective criteria.
        Also logs a summary of the best candidates found for each metric.
        """
        logger.info("Computing difference-in-means vectors...")
        difference_vectors = self.direction_finder_method.compute_difference_vectors(train_data, max_positions)

        logger.info("Pre-computing baseline scores and logits on the validation set...")
        positive_prompts = [p for p, label in zip(val_data.prompts, val_data.labels) if label]
        negative_prompts = [p for p, label in zip(val_data.prompts, val_data.labels) if not label]

        baseline_pos_logits = self.evaluator._get_logits(positive_prompts) if positive_prompts else []
        baseline_neg_logits = self.evaluator._get_logits(negative_prompts) if negative_prompts else []

        baseline_bypass_score = np.nanmean([self.evaluator.metric.compute_log_odds(logits) for logits in baseline_pos_logits]) if baseline_pos_logits else 0.0
        baseline_induce_score = np.nanmean([self.evaluator.metric.compute_log_odds(logits) for logits in baseline_neg_logits]) if baseline_neg_logits else 0.0

        logger.info(f"Baseline Scores | Bypass: {baseline_bypass_score:7.4f}, Induce: {baseline_induce_score:7.4f}, KL: 0.0")

        num_layers = len(self.intervention_applier.transformer_layers)
        layer_cutoff = int(self.search_config["layer_cutoff_frac"] * num_layers)

        logger.info(f"Evaluating direction candidates with multi-objective criteria...")

        all_scores_data = []
        best_overall_info = {'score': float('inf'), 'dir': None, 'scores': None}
        best_bypass_info = {'score': float('inf'), 'dir': None, 'scores': None}
        best_induce_info = {'score': float('-inf'), 'dir': None, 'scores': None}
        best_kl_info = {'score': float('inf'), 'dir': None, 'scores': None}

        selected_direction = None
        min_bypass_for_strict_selection = float('inf')

        candidate_iterator = tqdm(difference_vectors.items(), desc="Evaluating candidates")
        for (layer, pos_idx), vec in candidate_iterator:
            if layer >= layer_cutoff:
                continue

            current_direction = DirectionVector(vector=vec, layer=layer, position_index=pos_idx, score=0)
            scores = self.evaluator.compute_all_scores(current_direction, val_data, baseline_neg_logits=baseline_neg_logits)

            all_scores_data.append({
                'layer': layer,
                'position': pos_idx,
                'bypass_score': scores.bypass,
                'induce_score': scores.induce,
                'kl_score': scores.kl
            })

            lenient_score = (10 * scores.bypass) + scores.kl - scores.induce
            if lenient_score < best_overall_info['score']:
                best_overall_info.update({'score': lenient_score, 'dir': current_direction, 'scores': scores})
            if scores.bypass < best_bypass_info['score']:
                best_bypass_info.update({'score': scores.bypass, 'dir': current_direction, 'scores': scores})
            if scores.induce > best_induce_info['score']:
                best_induce_info.update({'score': scores.induce, 'dir': current_direction, 'scores': scores})
            if scores.kl < best_kl_info['score']:
                best_kl_info.update({'score': scores.kl, 'dir': current_direction, 'scores': scores})

            is_sufficient = scores.induce > self.search_config["induce_threshold"]
            is_safe = scores.kl < self.search_config["kl_threshold"]
            if is_sufficient and is_safe:
                if scores.bypass < min_bypass_for_strict_selection:
                    min_bypass_for_strict_selection = scores.bypass
                    current_direction.score = min_bypass_for_strict_selection
                    selected_direction = current_direction

        if all_scores_data:
            results_df = pd.DataFrame(all_scores_data)
            model_short_name = self.model.name_or_path.split('/')[-1] if hasattr(self.model, 'name_or_path') else 'unknown_model'
            # --- FIX: Call the plotting function ---
            self._plot_and_save_search_results(results_df, model_short_name)
        else:
            logger.warning("No data was collected during search; skipping data saving and plotting.")

        if selected_direction:
            logger.info(f"\n--- Strictly Selected Direction (Met All Criteria) ---")
            logger.info(f"Layer: {selected_direction.layer}, Position: {selected_direction.position_index}")
            logger.info(f"Final Bypass Score (minimized): {selected_direction.score:.4f}")
        else:
            logger.warning("\nNo direction vector was found that met all strict selection criteria. Printing additional information for debugging:")
            logger.info(f"  {'Baseline':<7}:               | Bypass: {baseline_bypass_score:7.4f}, Induce: {baseline_induce_score:7.4f}, KL: {0.0:7.4f}")
            self._print_debug_info("Overall", best_overall_info)
            self._print_debug_info("Bypass", best_bypass_info)
            self._print_debug_info("Induce", best_induce_info)
            self._print_debug_info("KL", best_kl_info)

        return selected_direction
