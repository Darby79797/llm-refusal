import torch as t
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import logging
import warnings

import lm_eval
from lm_eval.models.huggingface import HFLM

# Suppress a common warning from the harness about legacy constructors
warnings.filterwarnings("ignore", message="Using legacy validation features of the model repository")

from datatypes import DirectionVector
from interventions import ModelInterventionApplier
from formatting import ChatPromptFormatter
from concept import DEFAULT_REFUSAL_PHRASES

logger = logging.getLogger(__name__)


class InterventionSuite:
    """Runs qualitative tests on a given DirectionVector."""
    def __init__(self, model, tokenizer, intervention_applier, prompt_formatter):
        self.model = model
        self.tokenizer = tokenizer
        self.intervention_applier = intervention_applier
        self.prompt_formatter = prompt_formatter

    def test_generation(
        self,
        direction: DirectionVector,
        test_prompts: List[str],
        intervention_type: str = "ablate",
        strengths: List[float] = [1.0],
        max_new_tokens: int = 64
    ) -> Dict[str, List[Dict]]:
        """
        Tests interventions, attempting to use huggingface .generate().
        """
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False # Makes output deterministic, using greedy sampling.
        }
        results = {}
        batch_formatted = self.prompt_formatter.format_batch(test_prompts)
        input_ids = batch_formatted['input_ids'].to(self.model.device)
        attention_mask = batch_formatted['attention_mask'].to(self.model.device)

        # --- Baseline Generation ---
        logger.info(f"Generating baseline responses...")
        baseline_outputs = self.model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=self.tokenizer.eos_token_id
        )
        baseline_texts = self.tokenizer.batch_decode(baseline_outputs[:, input_ids.shape[1]:], skip_special_tokens=True)
        results["baseline_no_intervention"] = [{'prompt': p, 'generated_text': t} for p, t in zip(test_prompts, baseline_texts)]

        # --- Intervened Generation ---
        for strength in strengths:
            key = f"{intervention_type}_strength_{strength}"
            logger.info(f"Generating responses for intervention: {key}")

            if strength != 0.0 and direction is not None:
                layers = list(range(len(self.intervention_applier.transformer_layers))) if intervention_type == "ablate" else [direction.layer]
                self.intervention_applier.apply_direction_intervention(direction, intervention_type, strength, layers=layers)

            intervened_outputs = self.model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
            intervened_texts = self.tokenizer.batch_decode(intervened_outputs[:, input_ids.shape[1]:], skip_special_tokens=True)
            results[key] = [{'prompt': p, 'generated_text': t} for p, t in zip(test_prompts, intervened_texts)]

            self.intervention_applier.clear_interventions()

        return results


class BigEvaluator:
    """Handles quantitative evaluations for a given direction vector."""
    def __init__(self, framework: Any, detection_phrases: Optional[List[str]] = None):
        self.framework = framework
        self.model = framework.model
        self.tokenizer = framework.tokenizer
        self.intervention_applier = framework.intervention_applier
        self.detection_phrases = detection_phrases or DEFAULT_REFUSAL_PHRASES

    def _check_for_detection(self, text: str) -> bool:
        return any(phrase.lower() in text.lower() for phrase in self.detection_phrases)

    # Backward-compatible alias
    _check_for_refusal = _check_for_detection

    def evaluate_detection_rate(self, prompts: List[str], batch_size: int = 2) -> float:
        """
        Evaluates detection rate using batched .generate().
        """
        detection_count = 0
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Evaluating Detection Rate"):
            batch_prompts = prompts[i*batch_size : (i+1)*batch_size]
            batch_formatted = self.framework.prompt_formatter.format_batch(batch_prompts)
            input_ids = batch_formatted['input_ids'].to(self.model.device)
            attention_mask = batch_formatted['attention_mask'].to(self.model.device)

            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Slice off prompt tokens before decoding
            generated_texts = self.tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

            for text in generated_texts:
                if self._check_for_detection(text):
                    detection_count += 1

        return detection_count / len(prompts) if prompts else 0.0

    # Backward-compatible alias
    evaluate_refusal_rate = evaluate_detection_rate

    def run_standard_evals(self, eval_tasks: List[str], limit = 100) -> Dict[str, float]:
        """
        Runs the model on a specific list of standard benchmarks with robust device handling.
        """
        if len(eval_tasks)==0:
            logger.info(f"No tasks passed for evals. Skipping...")
            return {}

        logger.info(f"Starting standard evaluations for tasks: {eval_tasks}...")

        # HFLM expects a single device string ('cpu', 'mps', 'cuda:0') or None if using device_map.
        # If model.device is 'meta', it indicates the model is sharded across devices.
        device_str = None
        if self.model.device.type != 'meta':
            device_str = str(self.model.device)

        # HFLM may not like the hooked models and their interaction with parallelism.
        # options: (i) clone, (ii) single-GPU
        lm_eval_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            device=device_str, # Pass the corrected device string or None
        )

        # 3. Run the evaluation.
        if limit is None:
            limit = 100 # a default
        if limit != 'full':
            results = lm_eval.simple_evaluate(
                model=lm_eval_model,
                tasks=eval_tasks,
                batch_size="auto:4", # Automatically find best batch size, starting with 4
                log_samples=False,
                limit=limit
            )
        else: #limit == 'full'
            results = lm_eval.simple_evaluate(
                model=lm_eval_model,
                tasks=eval_tasks,
                batch_size="auto:4", # Automatically find best batch size, starting with 4
                log_samples=False
                # no limit of number of trials. Will be much slower.
            )

        logger.info("Standard evaluations complete. Parsing results...")

        # 4. Parse the results into a clean dictionary.
        scores = {}
        eval_results = results.get("results", {})

        # Extract the primary metric for each task.
        # lm-eval v0.4+ uses "metric,filter" keys (e.g. "acc,none").
        def _get(d, *keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        task_metric_map = {
            "mmlu": ("MMLU", ["acc,none", "acc"]),
            "arc_challenge": ("ARC-Challenge", ["acc_norm,none", "acc_norm"]),
            "gsm8k": ("GSM8K", ["acc,none", "exact_match,strict-match", "acc"]),
            "truthfulqa_mc2": ("TruthfulQA (MC2)", ["acc,none", "mc2"]),
        }
        for task_key, (display_name, metric_keys) in task_metric_map.items():
            if task_key in eval_results:
                val = _get(eval_results[task_key], *metric_keys)
                if val is not None:
                    scores[display_name] = val
        return scores

    def run_all_evaluations(self, direction: DirectionVector, positive_prompts: List[str], negative_prompts: List[str], tasks: List[str], limit: Optional[int]):
        """
        Orchestrates the entire evaluation suite. Now accepts tasks and limit.
        """
        results = {}
        num_layers = len(self.intervention_applier.transformer_layers)

        logger.info("\n--- Evaluating Baseline Model (No Interventions) ---")
        results["baseline"] = {
            "refusal_rate_on_positive_prompts": self.evaluate_detection_rate(positive_prompts),
            "standard_eval_scores": self.run_standard_evals(tasks, limit=limit)
        }

        logger.info("\n--- Evaluating Global Ablation (All Layers) ---")
        self.intervention_applier.apply_direction_intervention(direction, "ablate", 1.0, layers=list(range(num_layers)))
        results["global_ablation"] = {
            "refusal_rate_on_positive_prompts": self.evaluate_detection_rate(positive_prompts),
            "standard_eval_scores": self.run_standard_evals(tasks, limit=limit)
        }
        self.intervention_applier.clear_interventions()

        logger.info(f"\n--- Evaluating Layer-Specific Ablation (Layer {direction.layer}) ---")
        self.intervention_applier.apply_direction_intervention(direction, "ablate", 1.0, layers=[direction.layer])
        results["layer_specific_ablation"] = {
            "refusal_rate_on_positive_prompts": self.evaluate_detection_rate(positive_prompts),
            "standard_eval_scores": self.run_standard_evals(tasks, limit=limit)
        }
        self.intervention_applier.clear_interventions()

        logger.info(f"\n--- Evaluating Layer-Specific Addition (Layer {direction.layer}) ---")
        self.intervention_applier.apply_direction_intervention(direction, "add", 1.0, layers=[direction.layer])
        results["layer_specific_addition"] = {
            "refusal_rate_on_negative_prompts": self.evaluate_detection_rate(negative_prompts)
        }
        self.intervention_applier.clear_interventions()

        # --- Build the report as a string and log it ---
        report_lines = []
        for key, data in results.items():
            report_lines.append(f"\n--- {key.replace('_', ' ').upper()} ---")
            for metric, value in data.items():
                if isinstance(value, dict):
                    report_lines.append(f"  {metric}:")
                    for sub_metric, sub_value in value.items():
                        report_lines.append(f"    - {sub_metric}: {sub_value:.4f}")
                else:
                    report_lines.append(f"  {metric}: {value:.4f}")

        # Construct the final multi-line string
        header = "\n\n" + "="*20 + " COMPREHENSIVE EVALUATION REPORT " + "="*20
        report_body = "".join(report_lines)
        footer = "\n" + "="*70 + "\n"

        final_report = header + report_body + footer

        # Log the entire report string as a single info message
        logger.info(final_report)
