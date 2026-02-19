import pytest
import torch as t
import numpy as np
from unittest.mock import MagicMock

from datatypes import PromptData, DirectionVector
from formatting import ChatPromptFormatter
from interventions import ModelInterventionApplier
from activations import ActivationExtractor
from direction_methods import DifferenceInMeans
from scoring import LogOddsMetric, Three_Score_Evaluator
from framework import DirectionTestFramework
import warnings

def test_prompt_data_split(sample_prompt_data):
    """Tests that the train_val_split method works correctly."""
    train_data, val_data = sample_prompt_data.train_val_split(test_size=0.5, random_state=42)
    assert len(train_data.prompts) == 2
    assert len(val_data.prompts) == 2
    assert sum(train_data.labels) == 1
    assert sum(val_data.labels) == 1


# @pytest.mark.parametrize("model_name", [
#     "google/gemma-2b", # Base model
#     "meta-llama/Llama-3-8B-Instruct", # Instruct model with built-in template
#     "meta-llama/Llama-2-7b-chat-hf", # Chat model without built-in template
# ])
# def test_chat_prompt_formatter_logic(mock_tokenizer, model_name):
#     """
#     Tests the logic of ChatPromptFormatter by checking which tokenizer method is called.
#     """
#     tokenizer_mock = mock_tokenizer
#     tokenizer_mock.name_or_path = model_name
    
#     formatter = ChatPromptFormatter(tokenizer_mock)
#     prompts = ["test prompt"]
#     _ = formatter.format_batch(prompts) 

#     if formatter.is_instruction_tuned:
#         if tokenizer_mock.chat_template is not None:
#             # Asserts the built-in template was used
#             tokenizer_mock.apply_chat_template.assert_called_once()
#             tokenizer_mock.__call__.assert_not_called()
#         else:
#             # Asserts the manual template was used
#             expected_formatted_prompt = formatter.template.format(x=prompts[0])
#             tokenizer_mock.__call__.assert_called_once_with(
#                 [expected_formatted_prompt],
#                 padding=True, return_tensors="pt", truncation=True, max_length=formatter.safe_max_length
#             )
#             tokenizer_mock.apply_chat_template.assert_not_called()
#     else:
#         # Asserts no template was used (pass-through)
#         assert formatter.template == "{x}"
#         tokenizer_mock.__call__.assert_called_once_with(
#             prompts,
#             padding=True, return_tensors="pt", truncation=True, max_length=formatter.safe_max_length
#         )


def test_direction_vector_unit():
    """Tests the unit vector property of DirectionVector."""
    vec = t.tensor([3.0, 4.0])
    dv = DirectionVector(vector=vec, layer=0, position_index=-1, score=1.0)
    unit_vec = dv.unit
    assert t.allclose(t.norm(unit_vec), t.tensor(1.0))
    assert t.allclose(unit_vec, t.tensor([0.6, 0.8]))


def test_model_intervention_applier_hooks(mock_model):
    applier = ModelInterventionApplier(mock_model)
    direction = DirectionVector(t.randn(mock_model.config.hidden_size), 0, -1, 0.0)
    
    assert len(applier.intervention_hooks) == 0
    applier.apply_direction_intervention(direction, "add", layers=[0])
    
    mock_model.model.layers[0].register_forward_hook.assert_called_once()
    assert len(applier.intervention_hooks) == 1
    
    mock_hook = applier.intervention_hooks[0]
    applier.clear_interventions()
    mock_hook.remove.assert_called_once()
    assert len(applier.intervention_hooks) == 0

def test_difference_in_means(mocker, sample_prompt_data):
    mock_extractor = MagicMock(spec=ActivationExtractor)
    pos_activations = {(0, -1): t.ones(10)}
    neg_activations = {(0, -1): t.zeros(10)}
    mock_extractor.extract_residual_activations.side_effect = [pos_activations, neg_activations]
    
    dim = DifferenceInMeans(mock_extractor)
    diff_vectors = dim.compute_difference_vectors(sample_prompt_data)
    
    assert (0, -1) in diff_vectors
    assert t.allclose(diff_vectors[(0, -1)], t.ones(10))

def test_log_odds_metric(mock_tokenizer):
    # FIX: The fixture now returns a single mock tokenizer, no unpacking needed.
    tokenizer = mock_tokenizer
    target_tokens = ["target_A", "target_B"]
    metric = LogOddsMetric(tokenizer, target_tokens)
    
    assert len(metric.target_token_ids) == len(target_tokens)
    
    vocab_size = 50000
    logits = t.zeros(vocab_size)
    token_id_A, token_id_B = metric.target_token_ids.tolist()
    
    logits[token_id_A] = 5.0
    logits[token_id_B] = 5.0
    
    num_non_target = logits.shape[0] - 2
    expected = 5 + np.log(2) - np.log(num_non_target)
    assert abs(metric.compute_log_odds(logits) - expected) < 1e-4

def test_compute_all_scores_logic(mocker, mock_model, mock_tokenizer, mock_intervention_applier):
    # FIX: Pass the single mock_tokenizer and create the formatter mock inside the test.
    evaluator = Three_Score_Evaluator(mock_model, mock_tokenizer, mock_intervention_applier, MagicMock())

    mocker.patch.object(evaluator, '_compute_bypass_score', return_value=-1.25)
    mocker.patch.object(evaluator, '_compute_induce_score', return_value=0.75)
    mocker.patch.object(evaluator, '_compute_kl_score', return_value=0.05)

    scores = evaluator.compute_all_scores(MagicMock(), MagicMock())

    assert scores.bypass == -1.25
    assert scores.induce == 0.75
    assert scores.kl == 0.05

def test_compute_bypass_score_logic(mocker, sample_prompt_data, mock_model, mock_tokenizer, mock_intervention_applier):
    evaluator = Three_Score_Evaluator(mock_model, mock_tokenizer, mock_intervention_applier, MagicMock())
    
    mocker.patch.object(evaluator, '_get_logits', return_value=[t.randn(10), t.randn(10)])
    evaluator.metric = MagicMock(spec=LogOddsMetric)
    evaluator.metric.compute_log_odds.side_effect = [-1.0, -1.5]

    bypass_score = evaluator._compute_bypass_score(MagicMock(), sample_prompt_data)
    assert bypass_score == pytest.approx(-1.25)

def test_compute_induce_score_logic(mocker, sample_prompt_data, mock_model, mock_tokenizer, mock_intervention_applier):
    evaluator = Three_Score_Evaluator(mock_model, mock_tokenizer, mock_intervention_applier, MagicMock())
    
    mocker.patch.object(evaluator, '_get_logits', return_value=[t.randn(10), t.randn(10)])
    evaluator.metric = MagicMock(spec=LogOddsMetric)
    evaluator.metric.compute_log_odds.side_effect = [0.5, 1.0]

    mock_direction = MagicMock(spec=DirectionVector, layer=1)
    induce_score = evaluator._compute_induce_score(mock_direction, sample_prompt_data)
    assert induce_score == pytest.approx(0.75)

def test_compute_kl_score_logic(mocker, sample_prompt_data, mock_model, mock_tokenizer, mock_intervention_applier):
    evaluator = Three_Score_Evaluator(mock_model, mock_tokenizer, mock_intervention_applier, MagicMock())

    mocker.patch.object(evaluator, '_get_logits', return_value=[t.randn(10), t.randn(10)])
    mocker.patch('scoring.F.kl_div', return_value=t.tensor(0.05))

    kl_score = evaluator._compute_kl_score(MagicMock(), sample_prompt_data)
    assert kl_score == pytest.approx(0.05)