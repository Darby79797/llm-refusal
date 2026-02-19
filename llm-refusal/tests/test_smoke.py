import pytest
import torch as t
import numpy as np
import os

from framework import DirectionTestFramework
from datatypes import PromptData, DirectionVector

@pytest.mark.smoke
def test_full_pipeline_smoke(real_tiny_model_and_tokenizer):
    """
    Runs a lightweight, end-to-end test of the full pipeline to ensure the
    environment is set up correctly and the code can execute without errors.
    This test is designed to run quickly (< 2 mins).
    """
    
    # --- 1. Setup ---
    model, tokenizer = real_tiny_model_and_tokenizer
    
    # We will instantiate the framework but point it to our tiny model
    # To do this, we patch the __init__ method to use our fixtures
    # This is a bit of a hack, but avoids needing a separate "testing" config
    class PatchedFramework(DirectionTestFramework):
        def __init__(self, *args, **kwargs):
            self.model_name = "sshleifer/tiny-gpt2"
            self.model = model
            self.tokenizer = tokenizer
            self.device = model.device

            # Manually instantiate dependencies
            from interventions import ModelInterventionApplier
            from formatting import ChatPromptFormatter
            from activations import ActivationExtractor
            from direction_methods import DifferenceInMeans
            from scoring import Three_Score_Evaluator
            from evaluation import BigEvaluator, InterventionSuite
            from concept import DEFAULT_REFUSAL_TOKENS, DEFAULT_REFUSAL_PHRASES
            self.intervention_applier = ModelInterventionApplier(self.model)
            self.prompt_formatter = ChatPromptFormatter(self.tokenizer)
            self.extractor = ActivationExtractor(self.model, self.tokenizer, self.intervention_applier.transformer_layers, self.prompt_formatter)
            self.direction_finder = DifferenceInMeans(self.extractor)
            self.cheap_evaluator = Three_Score_Evaluator(self.model, self.tokenizer, self.intervention_applier, self.prompt_formatter, target_tokens=DEFAULT_REFUSAL_TOKENS)
            self.big_evaluator = BigEvaluator(self, detection_phrases=DEFAULT_REFUSAL_PHRASES)
            self.suite = InterventionSuite(self.model, self.tokenizer, self.intervention_applier, self.prompt_formatter)

    framework = PatchedFramework()

    # Use a tiny dataset for speed
    train_data = PromptData(
        prompts=["Positive prompt 1", "Positive prompt 2", "Negative prompt 1", "Negative prompt 2"],
        labels=[True, True, False, False]
    )
    val_data = PromptData(
        prompts=["Pos val 1", "Pos val 2", "Neg val 1", "Neg val 2"],
        labels=[True, True, False, False]
    )

    # --- 2. Execution ---
    try:
        # A. Find a direction vector
        difference_vectors = framework.direction_finder.compute_difference_vectors(train_data, max_positions=2)
        assert len(difference_vectors) > 0, "Difference-in-means failed to produce any vectors."
        
        # Select the first available vector for testing
        (layer, pos_idx), vec = next(iter(difference_vectors.items()))
        direction = DirectionVector(vector=vec, layer=layer, position_index=pos_idx, score=0.0)

        # B. Evaluate the vector
        scores = framework.cheap_evaluator.compute_all_scores(direction, val_data)

        # C. Run a single intervention test
        # We'll skip the full eval suite as it's slow, but test one intervention.
        test_prompts = ["This is a test prompt for intervention."]
        results = framework.suite.test_generation(direction, test_prompts, intervention_type="add", strengths=[1.0])
        
    except Exception as e:
        # Fail the test for any other unexpected exception.
        pytest.fail(f"Smoke test failed with an unexpected exception: {e}")

    # --- 3. Assertions ---
    # We don't check for specific values, just that the pipeline ran and produced
    # outputs of the correct type and within a reasonable range.
    assert isinstance(scores.bypass, float) and np.isfinite(scores.bypass)
    assert isinstance(scores.induce, float) and np.isfinite(scores.induce)
    assert isinstance(scores.kl, float) and np.isfinite(scores.kl) and scores.kl >= 0

    assert "baseline_no_intervention" in results
    assert "add_strength_1.0" in results
    assert len(results["baseline_no_intervention"]) == 1
    assert "generated_text" in results["baseline_no_intervention"][0]

    print("\nSmoke test passed: Pipeline executed successfully.")