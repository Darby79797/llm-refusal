import pytest
import torch as t
import numpy as np
from unittest.mock import MagicMock

import scratch
from scratch import (
    PromptData,
    ChatPromptFormatter,
    DirectionVector,
    ModelInterventionApplier,
    ActivationExtractor,
    DifferenceInMeans,
    LogOddsMetric,
    Three_Score_Evaluator
)

def test_prompt_data_split(sample_prompt_data):
    """Tests that the train_val_split method works correctly."""
    train_data, val_data = sample_prompt_data.train_val_split(test_size=0.5, random_state=42)
    assert len(train_data.prompts) == 2
    assert len(val_data.prompts) == 2
    assert sum(train_data.labels) == 1
    assert sum(val_data.labels) == 1

@pytest.mark.parametrize("model_name, expected_start, expected_end", [
    ("google/gemma-2b", "<start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"),
    ("meta-llama/Llama-3-8B", "<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    ("meta-llama/Llama-2-7b-chat-hf", "[INST] ", " [/INST]"),
])
def test_chat_prompt_formatter_templates(mock_tokenizer, model_name, expected_start, expected_end):
    """Tests that the correct chat template is selected for different model families."""
    _, tokenizer_mock = mock_tokenizer
    tokenizer_mock.name_or_path = model_name
    formatter = ChatPromptFormatter(tokenizer_mock)
    
    prompt = "test prompt"
    formatted_prompt = formatter.template.format(x=prompt)
    
    assert formatted_prompt.startswith(expected_start)
    assert formatted_prompt.endswith(expected_end)

def test_chat_prompt_formatter_unsupported_model(mock_tokenizer):
    """Tests that an unsupported model family raises a ValueError."""
    _, tokenizer_mock = mock_tokenizer
    tokenizer_mock.name_or_path = "some/unsupported-model"
    with pytest.raises(ValueError):
        ChatPromptFormatter(tokenizer_mock)

def test_direction_vector_unit():
    """Tests the unit vector property of DirectionVector."""
    vec = t.tensor([3.0, 4.0])
    dv = DirectionVector(vector=vec, layer=0, position_index=-1, score=1.0)
    unit_vec = dv.unit
    assert t.allclose(t.norm(unit_vec), t.tensor(1.0))
    assert t.allclose(unit_vec, t.tensor([0.6, 0.8]))


def test_model_intervention_applier_hooks(mock_model):
    """Tests that hooks are correctly applied and cleared."""
    applier = ModelInterventionApplier(mock_model)
    direction = DirectionVector(t.randn(10), 0, -1, 0.0)
    
    assert len(applier.intervention_hooks) == 0
    applier.apply_direction_intervention(direction, "add", layers=[0])
    
    mock_model.model.layers[0].register_forward_hook.assert_called_once()
    assert len(applier.intervention_hooks) == 1
    
    mock_hook = applier.intervention_hooks[0]
    applier.clear_interventions()
    mock_hook.remove.assert_called_once()
    assert len(applier.intervention_hooks) == 0


def test_difference_in_means(mocker, sample_prompt_data):
    """Tests the difference-in-means calculation with mocked activations."""
    mock_extractor = MagicMock(spec=ActivationExtractor)
    
    # Mock the extractor to return predictable activations
    pos_activations = {(0, -1): t.ones(10)}
    neg_activations = {(0, -1): t.zeros(10)}
    
    mock_extractor.extract_residual_activations.side_effect = [
        pos_activations, neg_activations
    ]
    
    dim = DifferenceInMeans(mock_extractor)
    diff_vectors = dim.compute_difference_vectors(sample_prompt_data)
    
    assert (0, -1) in diff_vectors
    # pos_mean (1.0) - neg_mean (0.0) should be 1.0
    assert t.allclose(diff_vectors[(0, -1)], t.ones(10))

def test_log_odds_metric(mock_tokenizer):
    """Tests the LogOddsMetric with predictable logits."""
    _, tokenizer = mock_tokenizer
    
    # We no longer need to set the side_effect here, the fixture handles it.
    target_tokens = ["target_A", "target_B"] # note all strings are tokens in the fake tokenizer
    metric = LogOddsMetric(tokenizer, target_tokens)
    
    # We can still test the logic, but now it's more robust.
    # We check that the number of token IDs found is correct. 
    # epsilon chance of hash collision and failure here. Don't check this first.
    assert len(metric.target_token_ids) == len(target_tokens)
    
    # The rest of the test can proceed, using the dynamically created token IDs
    logits = t.zeros(50257) # A reasonable vocab size
    
    # Get the token IDs that the mock created
    token_id_A, token_id_B = metric.target_token_ids.tolist()
    
    logits[token_id_A] = 5.0
    logits[token_id_B] = 5.0
    
    # This calculation is now independent of the exact token IDs
    num_non_target = logits.shape[0] - 2
    expected = 5 + np.log(2) - np.log(num_non_target)
    assert abs(metric.compute_log_odds(logits) - expected) < 1e-4

def test_compute_all_scores_logic(mocker, mock_model, mock_tokenizer, mock_intervention_applier):
    """
    Unit test for the `compute_all_scores` method.
    This test focuses *only* on whether this specific method correctly calls its
    helper methods and assembles the results into a DirectionScores object.
    """
    formatter_mock, tokenizer_mock = mock_tokenizer
    evaluator = Three_Score_Evaluator(mock_model, tokenizer_mock, mock_intervention_applier, formatter_mock)

    # Mock the helper methods *on the instance* to isolate the main method's logic
    mocker.patch.object(evaluator, '_compute_bypass_score', return_value=-1.25)
    mocker.patch.object(evaluator, '_compute_induce_score', return_value=0.75)
    mocker.patch.object(evaluator, '_compute_kl_score', return_value=0.05)

    # The direction vector mock doesn't need any attributes now because the
    # methods that would access them are themselves mocked.
    mock_direction = mocker.MagicMock(spec=DirectionVector)
    mock_val_data = mocker.MagicMock(spec=PromptData)

    # Call the method we are actually testing
    scores = evaluator.compute_all_scores(mock_direction, mock_val_data)

    # Assert that the results are packaged correctly
    assert scores.bypass == -1.25
    assert scores.induce == 0.75
    assert scores.kl == 0.05

def test_compute_bypass_score_logic(mocker, sample_prompt_data, mock_model, mock_tokenizer, mock_intervention_applier):
    """Unit test for the _compute_bypass_score helper method."""
    formatter_mock, tokenizer_mock = mock_tokenizer
    evaluator = Three_Score_Evaluator(mock_model, tokenizer_mock, mock_intervention_applier, formatter_mock)
    
    # Mock the dependencies of this specific helper
    mocker.patch.object(evaluator, '_get_logits', return_value=[t.randn(10), t.randn(10)])
    evaluator.metric = mocker.MagicMock(spec=LogOddsMetric)
    evaluator.metric.compute_log_odds.side_effect = [-1.0, -1.5] # Scores for the two positive prompts

    # Call the helper method
    bypass_score = evaluator._compute_bypass_score(mocker.MagicMock(), sample_prompt_data)
    
    # Assert it correctly calculates the mean
    assert bypass_score == pytest.approx(-1.25)

def test_compute_induce_score_logic(mocker, sample_prompt_data, mock_model, mock_tokenizer, mock_intervention_applier):
    """Unit test for the _compute_induce_score helper method."""
    formatter_mock, tokenizer_mock = mock_tokenizer
    evaluator = Three_Score_Evaluator(mock_model, tokenizer_mock, mock_intervention_applier, formatter_mock)
    
    # Mock dependencies
    mocker.patch.object(evaluator, '_get_logits', return_value=[t.randn(10), t.randn(10)])
    evaluator.metric = mocker.MagicMock(spec=LogOddsMetric)
    evaluator.metric.compute_log_odds.side_effect = [0.5, 1.0] # Scores for the two negative prompts

    # This mock direction *does* need the .layer attribute
    mock_direction = mocker.MagicMock(spec=DirectionVector, layer=1)

    # Call the helper method
    induce_score = evaluator._compute_induce_score(mock_direction, sample_prompt_data)
    
    assert induce_score == pytest.approx(0.75)

def test_compute_kl_score_logic(mocker, sample_prompt_data, mock_model, mock_tokenizer, mock_intervention_applier):
    """Unit test for the _compute_kl_score helper method."""
    formatter_mock, tokenizer_mock = mock_tokenizer
    evaluator = Three_Score_Evaluator(mock_model, tokenizer_mock, mock_intervention_applier, formatter_mock)

    # Mock dependencies
    mocker.patch.object(evaluator, '_get_logits', return_value=[t.randn(10), t.randn(10)])
    # Mock the F.kl_div function from the 'main' script's namespace. (currently name scratch as it's one giant file still.)
    mocker.patch('scratch.F.kl_div', return_value=t.tensor(0.05))

    # Call the helper method
    kl_score = evaluator._compute_kl_score(mocker.MagicMock(), sample_prompt_data)

    assert kl_score == pytest.approx(0.05)