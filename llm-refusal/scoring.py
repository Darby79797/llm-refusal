import contextlib
import numpy as np
import torch as t
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import logging

from datatypes import DirectionVector, DirectionScores, PromptData
from interventions import ModelInterventionApplier
from formatting import ChatPromptFormatter, last_real_token_indices
from concept import DEFAULT_REFUSAL_TOKENS

logger = logging.getLogger(__name__)


class DirectionEvaluator(ABC):
    """Abstract base class for evaluating the quality of a direction vector."""

    @abstractmethod
    def evaluate_direction(
        self,
        direction_vector: DirectionVector,
        val_data: PromptData
    ) -> float:
        """Evaluate how well a direction vector captures the target concept."""
        pass


class LogOddsMetric:
    """
    A metric for calculating the log-odds of target tokens from model logits.
    Includes robust token ID validation and a safety check for nan/inf logits.
    """
    def __init__(self, tokenizer, target_tokens: List[str]):
        self.tokenizer = tokenizer
        self.target_token_ids = []
        for token in target_tokens:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) == 1:
                self.target_token_ids.append(encoded[0])
        if not self.target_token_ids:
            raise ValueError("CRITICAL: No valid target tokens were found.")
        self.target_token_ids = t.tensor(self.target_token_ids, dtype=t.long)

    def compute_log_odds(self, logits: t.Tensor) -> float:
        """Computes log(P(target) / P(not_target)) in a direct and stable way."""
        if t.isinf(logits).any() or t.isnan(logits).any():
            logger.warning("Logits tensor contains 'inf' or 'nan' values. This will result in a nan score.")
            return float('nan')

        device_target_ids = self.target_token_ids.to(logits.device)
        target_log_sum_exp = t.logsumexp(logits[device_target_ids], dim=0)
        mask = t.ones_like(logits, dtype=t.bool)
        mask[device_target_ids] = False
        non_target_log_sum_exp = t.logsumexp(logits[mask], dim=0)
        return (target_log_sum_exp - non_target_log_sum_exp).item()


class Three_Score_Evaluator:
    """Evaluates direction vectors using bypass, induce, and KL divergence scores."""
    def __init__(self, model, tokenizer, intervention_applier: ModelInterventionApplier, prompt_formatter: ChatPromptFormatter, target_tokens: Optional[List[str]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.intervention_applier = intervention_applier
        self.prompt_formatter = prompt_formatter
        self.metric = LogOddsMetric(tokenizer, target_tokens or DEFAULT_REFUSAL_TOKENS)
        self.device = model.device

    def _get_logits(self, prompts: List[str], intervention: Optional[Tuple] = None) -> List[t.Tensor]:
        all_logits = []
        if intervention:
            direction, int_type, layers = intervention
            ctx = self.intervention_applier.intervened(direction, int_type, strength=1.0, layers=layers)
        else:
            ctx = contextlib.nullcontext()
        with ctx, t.no_grad():
            batch = self.prompt_formatter.format_batch(prompts)
            input_ids, attention_mask = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
            position_ids = batch['position_ids'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            last_token_indices = last_real_token_indices(attention_mask)
            batch_logits = outputs.logits[t.arange(outputs.logits.size(0)), last_token_indices, :]
            if t.isinf(batch_logits).any() or t.isnan(batch_logits).any():
                problem_indices = t.nonzero(t.isinf(batch_logits).any(dim=1) | t.isnan(batch_logits).any(dim=1)).squeeze().tolist()
                if not isinstance(problem_indices, list): problem_indices = [problem_indices]
                for idx in problem_indices: logger.warning(f"NaN/Inf detected in logits for prompt at batch index {idx}: '{prompts[idx][:100]}...'")
            # Upcast before any softmax/KL: bf16 models emit bf16 logits, and
            # log_softmax/KL over a ~150k vocab in bf16 gives e.g. a nonzero
            # (even negative) self-KL. Small (~0.001 KL, ~0.02 log-odds on
            # Qwen2.5-1.5B) but free to remove.
            all_logits = [logit.float().cpu() for logit in batch_logits]
        return all_logits

    def _compute_bypass_score(self, direction: DirectionVector, val_data: PromptData) -> float:
        positive_prompts = val_data.positive
        if not positive_prompts: return 0.0
        num_layers = len(self.intervention_applier.transformer_layers)
        logits_with_ablation = self._get_logits(positive_prompts, intervention=(direction, "ablate", list(range(num_layers))))
        return np.nanmean([self.metric.compute_log_odds(logits) for logits in logits_with_ablation])

    def _compute_induce_score(self, direction: DirectionVector, val_data: PromptData) -> float:
        negative_prompts = val_data.negative
        if not negative_prompts: return 0.0
        logits_with_addition = self._get_logits(negative_prompts, intervention=(direction, "add", [direction.layer]))
        return np.nanmean([self.metric.compute_log_odds(logits) for logits in logits_with_addition])

    def _compute_induce_score_global(self, direction: DirectionVector, val_data: PromptData) -> float:
        """Induce score with direction added at ALL layers (Arditi-style)."""
        negative_prompts = val_data.negative
        if not negative_prompts: return 0.0
        num_layers = len(self.intervention_applier.transformer_layers)
        logits = self._get_logits(negative_prompts, intervention=(direction, "add", list(range(num_layers))))
        return np.nanmean([self.metric.compute_log_odds(l) for l in logits])

    def _compute_kl_score(self, direction: DirectionVector, val_data: PromptData, baseline_logits: Optional[List[t.Tensor]] = None) -> float:
        negative_prompts = val_data.negative
        if not negative_prompts: return 0.0
        num_layers = len(self.intervention_applier.transformer_layers)
        if baseline_logits is None:
            logger.warning("Re-computing baseline logits inside KL score. For performance, pre-compute and pass them.")
            baseline_logits = self._get_logits(negative_prompts)
        ablated_logits = self._get_logits(negative_prompts, intervention=(direction, "ablate", list(range(num_layers))))
        kl_divergences = [F.kl_div(F.log_softmax(ablated_logit, dim=-1), F.softmax(baseline_logit, dim=-1), reduction='sum', log_target=False).item() for baseline_logit, ablated_logit in zip(baseline_logits, ablated_logits)]
        return np.nanmean(kl_divergences)

    def compute_all_scores(self, direction_vector: DirectionVector, val_data: PromptData, baseline_neg_logits: Optional[List[t.Tensor]] = None) -> DirectionScores:
        bypass_score = self._compute_bypass_score(direction_vector, val_data)
        induce_score = self._compute_induce_score(direction_vector, val_data)
        induce_global = self._compute_induce_score_global(direction_vector, val_data)
        kl_score = self._compute_kl_score(direction_vector, val_data, baseline_logits=baseline_neg_logits)
        return DirectionScores(bypass=bypass_score, induce=induce_score, kl=kl_score, induce_global=induce_global)
