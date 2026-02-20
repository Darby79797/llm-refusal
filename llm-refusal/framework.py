import torch as t
from typing import Dict, List, Union, Optional
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from datatypes import PromptData, DirectionVector
from concept import get_concept
from formatting import ChatPromptFormatter
from activations import ActivationExtractor
from interventions import ModelInterventionApplier
from direction_methods import DifferenceInMeans
from scoring import Three_Score_Evaluator
from search import DirectionFinder
from evaluation import InterventionSuite, BigEvaluator

logger = logging.getLogger(__name__)


class DirectionTestFramework:
    """
    Main orchestrator for finding, evaluating, and testing direction vectors.
    """
    def __init__(self, model_name: str, torch_dtype: Union[str, t.dtype] = "auto", force_cpu: bool = False, concept: str = "refusal"):
        self.model_name = model_name
        self.concept = get_concept(concept)

        if force_cpu:
            self.device = t.device("cpu")
            logger.warning("CPU has been forced for model execution.")
        elif t.cuda.is_available():
            self.device = t.device("cuda:0")
            logger.info(f"CUDA found. Using single GPU ({self.device}) for execution.")
        elif t.backends.mps.is_available():
            self.device = t.device("mps")
            logger.info("MPS device found. Using MPS for model.")
        else:
            self.device = t.device("cpu")
            logger.info(f"No GPU/MPS found. Using device: {self.device}")
        logger.info(f"Loading chat model '{model_name}' to device '{self.device}'...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device  # Use device_map instead of .to(). Not sure this is actually necessary.
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # --- MPS precision safety ---
        # float16 on MPS causes attention overflow (NaN/Inf). Upcast to bfloat16.
        if self.device.type == "mps" and self.model.dtype == t.float16:
            logger.warning("Model loaded as float16 on MPS — upcasting to bfloat16 to prevent attention overflow.")
            self.model = self.model.to(dtype=t.bfloat16)

        # Initialize the modular components
        self.intervention_applier = ModelInterventionApplier(self.model)
        self.prompt_formatter = ChatPromptFormatter(self.tokenizer)

        # Wire concept through to components
        extractor = ActivationExtractor(self.model, self.tokenizer, self.intervention_applier.transformer_layers, self.prompt_formatter)
        direction_method = DifferenceInMeans(extractor)
        evaluator = Three_Score_Evaluator(self.model, self.tokenizer, self.intervention_applier, self.prompt_formatter, target_tokens=self.concept.target_tokens)

        self.finder = DirectionFinder(
            self.model, self.tokenizer, self.intervention_applier, self.prompt_formatter,
            direction_method=direction_method,
            evaluator=evaluator,
            search_config=self.concept.search_config
        )
        self.suite = InterventionSuite(self.model, self.tokenizer, self.intervention_applier, self.prompt_formatter)
        self.evaluator = BigEvaluator(self, detection_phrases=self.concept.detection_phrases,
                                      detection_fn=self.concept.detection_fn,
                                      judge_prompt=self.concept.judge_prompt)

        logger.info(f"Framework initialized on device: {self.device}")

    def _print_eyeball_results(self, results: Dict[str, List[Dict]]):
        """Pretty-prints the results from InterventionSuite.test_generation()."""
        for condition, entries in results.items():
            logger.info(f"\n  [{condition}]")
            for entry in entries:
                prompt_short = entry['prompt'][:80]
                generated = entry['generated_text'][:200]
                logger.info(f"    Prompt:   {prompt_short}")
                logger.info(f"    Response: {generated}")
                logger.info("")

    def run(self, config: Dict):
        """
        Main execution method based on the provided config.
        """
        positive_prompts, negative_prompts = self.concept.train_data_fn()
        train_data = PromptData(positive_prompts + negative_prompts, [True]*len(positive_prompts) + [False]*len(negative_prompts))
        train_data, val_data = train_data.train_val_split()

        eval_pos, eval_neg = self.concept.eval_data_fn()

        direction_to_test = None

        if config['mode'] == "search":
            logger.info("Running in SEARCH mode...")
            direction_to_test = self.finder.find_best_direction(train_data, val_data)
            if direction_to_test is None:
                logger.error("Search concluded without finding a suitable direction vector.")
                return

        elif config['mode'] in ["eyeball", "evaluate"]:
            layer, pos = config.get('layer'), config.get('pos')
            if layer is None or pos is None:
                logger.error(f"Mode '{config['mode']}' requires 'layer' and 'pos' to be specified.")
                return

            logger.info(f"Using pre-specified vector for Layer {layer}, Position {pos}...")
            # We still need to compute the vector, even if we know the location
            diff_vectors = self.finder.direction_finder_method.compute_difference_vectors(train_data)
            vec = diff_vectors.get((layer, pos))
            if vec is None:
                logger.error(f"Vector at ({layer}, {pos}) not found. Exiting.")
                return
            direction_to_test = DirectionVector(vector=vec, layer=layer, position_index=pos, score=0.0)

        if direction_to_test is None:
            logger.warning("No direction vector to test. Exiting.")
            return

        # --- Now, run the appropriate suite based on the mode ---
        if config['mode'] == "eyeball":
            logger.info("\n--- Running Eyeball Tests on Positive Prompts (Ablation) ---")
            results_ablate = self.suite.test_generation(direction_to_test, eval_pos[:5], "ablate")
            self._print_eyeball_results(results_ablate)

            logger.info("\n--- Running Eyeball Tests on Negative Prompts (Addition) ---")
            results_add = self.suite.test_generation(direction_to_test, eval_neg[:5], "add")
            self._print_eyeball_results(results_add)

        elif config['mode'] == "evaluate":
            logger.info("\n--- Running Full Evaluation Suite ---")
            self.evaluator.run_all_evaluations(
                direction_to_test, eval_pos, eval_neg,
                tasks=config.get('eval_tasks', []),
                limit=config.get('limit', 100)
            )

        logger.info("Framework execution finished.")


def main(
    config: Dict
):
    """
    Main execution function with multiple modes: search, evaluate, or eyeball
    """
    framework = DirectionTestFramework(
        model_name=config['model_name'],
        torch_dtype=config['torch_dtype'],
        force_cpu=config['force_cpu'],
        concept=config.get('concept', 'refusal')
    )
    framework.run(config)
