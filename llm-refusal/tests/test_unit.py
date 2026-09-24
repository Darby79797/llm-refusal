import pytest
import json as json_mod
import torch as t
import numpy as np
from unittest.mock import MagicMock

from datatypes import PromptData, DirectionVector, DirectionScores
from scoring import LogOddsMetric
from evaluation import BigEvaluator

import warnings


# --- Helper ---

def _make_evaluator(**kwargs):
    """Helper to create a BigEvaluator with a mock framework (no spec constraints)."""
    framework = MagicMock()
    framework.model.device = t.device("cpu")
    framework.intervention_applier.transformer_layers = []
    return BigEvaluator(framework, **kwargs)


# --- SimpleTokenizer for LogOddsMetric tests ---

class SimpleTokenizer:
    """Minimal tokenizer that maps known tokens to fixed IDs."""
    def __init__(self, token_map=None):
        self._map = token_map or {}

    def encode(self, token, add_special_tokens=False):
        if token in self._map:
            return [self._map[token]]
        return [hash(token) % 50000]


# ============================================================
# Data & datatypes
# ============================================================

def test_prompt_data_split(sample_prompt_data):
    """Tests that the train_val_split method works correctly."""
    train_data, val_data = sample_prompt_data.train_val_split(test_size=0.5, random_state=42)
    assert len(train_data.prompts) == 2
    assert len(val_data.prompts) == 2
    assert sum(train_data.labels) == 1
    assert sum(val_data.labels) == 1


def test_prompt_data_split_preserves_all_prompts():
    """train_val_split is a complete partition: no data lost or duplicated."""
    data = PromptData(prompts=[f"p{i}" for i in range(20)],
                      labels=[i < 10 for i in range(20)])
    train, val = data.train_val_split(test_size=0.3, random_state=42)
    all_prompts = set(train.prompts + val.prompts)
    assert len(all_prompts) == 20
    assert len(train.prompts) + len(val.prompts) == 20


def test_direction_vector_unit():
    """Tests the unit vector property of DirectionVector."""
    vec = t.tensor([3.0, 4.0])
    dv = DirectionVector(vector=vec, layer=0, position_index=-1, score=1.0)
    unit_vec = dv.unit
    assert t.allclose(t.norm(unit_vec), t.tensor(1.0))
    assert t.allclose(unit_vec, t.tensor([0.6, 0.8]))


def test_direction_vector_unit_zero_vector():
    """Unit vector of zero vector produces NaN (divide-by-zero)."""
    dv = DirectionVector(vector=t.zeros(10), layer=0, position_index=-1, score=0.0)
    assert t.isnan(dv.unit).all()


# ============================================================
# Concept registry
# ============================================================

def test_concept_registry_get_registered():
    """get_concept returns valid ConceptDefinitions for all registered concepts."""
    from concept import get_concept, CONCEPT_REGISTRY
    for name in CONCEPT_REGISTRY:
        concept = get_concept(name)
        assert concept.name == name
        assert callable(concept.train_data_fn)
        assert callable(concept.eval_data_fn)
        assert len(concept.target_tokens) > 0


def test_concept_registry_get_unregistered():
    """get_concept raises KeyError for unknown names."""
    from concept import get_concept
    with pytest.raises(KeyError):
        get_concept("nonexistent_concept_xyz")


# ============================================================
# LogOddsMetric
# ============================================================

def test_log_odds_metric():
    """LogOddsMetric computes correct log-odds for known target tokens."""
    tokenizer = SimpleTokenizer({"target_A": 100, "target_B": 200})
    metric = LogOddsMetric(tokenizer, ["target_A", "target_B"])

    assert len(metric.target_token_ids) == 2

    vocab_size = 50000
    logits = t.zeros(vocab_size)
    logits[100] = 5.0
    logits[200] = 5.0

    result = metric.compute_log_odds(logits)
    # log(exp(5)+exp(5)) - log(sum of rest) = 5 + log(2) - log(49998)
    expected = 5 + np.log(2) - np.log(49998)
    assert abs(result - expected) < 1e-4


def test_log_odds_metric_nan_handling():
    """Returns NaN for logits containing inf or nan."""
    tokenizer = SimpleTokenizer({"x": 42})
    metric = LogOddsMetric(tokenizer, ["x"])

    logits_inf = t.zeros(1000)
    logits_inf[0] = float('inf')
    assert np.isnan(metric.compute_log_odds(logits_inf))

    logits_nan = t.zeros(1000)
    logits_nan[0] = float('nan')
    assert np.isnan(metric.compute_log_odds(logits_nan))


def test_log_odds_metric_no_valid_tokens_raises():
    """Raises ValueError when no target tokens produce single-token encodings."""
    class MultiTokenizer:
        def encode(self, token, add_special_tokens=False):
            return [1, 2, 3]  # Always multi-token
    with pytest.raises(ValueError, match="No valid target tokens"):
        LogOddsMetric(MultiTokenizer(), ["any_token"])


# ============================================================
# Search fallback logic
# ============================================================

def test_search_progressive_fallback_tier_selection():
    """Tier 1 (delta>+3, KL<5.0) is selected when a candidate qualifies."""
    from search import DirectionFinder

    finder = DirectionFinder.__new__(DirectionFinder)
    candidates = [
        (DirectionVector(vector=t.randn(32), layer=5, position_index=-1, score=0),
         DirectionScores(bypass=-1.0, induce=4.5, kl=3.0)),   # passes tier 1 (delta=+3.5, KL<5)
        (DirectionVector(vector=t.randn(32), layer=8, position_index=-1, score=0),
         DirectionScores(bypass=-0.5, induce=5.0, kl=6.0)),   # fails tier 1 (KL>5.0)
    ]
    result = finder._progressive_fallback(candidates, baseline_induce=1.0)
    assert result.layer == 5  # Only tier-1-passing candidate


def test_search_fallback_best_induce_last_resort():
    """When no candidates pass any tier, falls back to best induce."""
    from search import DirectionFinder

    finder = DirectionFinder.__new__(DirectionFinder)
    candidates = [
        (DirectionVector(vector=t.randn(32), layer=3, position_index=-1, score=0),
         DirectionScores(bypass=-1.0, induce=0.5, kl=50.0)),
        (DirectionVector(vector=t.randn(32), layer=7, position_index=-1, score=0),
         DirectionScores(bypass=-2.0, induce=0.8, kl=50.0)),
    ]
    result = finder._progressive_fallback(candidates, baseline_induce=10.0)
    assert result.layer == 7  # Best induce (0.8 > 0.5)


# ============================================================
# Sycophancy detection heuristic
# ============================================================

from concept import detect_sycophancy


def test_detect_sycophancy_clear_agreement():
    """Pure agreement without contradiction is sycophantic."""
    text = "You're absolutely right! That's a great observation."
    assert detect_sycophancy(text) is True


def test_detect_sycophancy_agreement_then_contradiction():
    """Agreement followed by early contradiction is NOT sycophantic."""
    text = "That's a great point, however, the scientific evidence actually shows the opposite."
    assert detect_sycophancy(text) is False


def test_detect_sycophancy_clear_correction():
    """Direct correction with no agreement is NOT sycophantic."""
    text = "Actually, that's a common misconception. The Great Wall is not visible from space."
    assert detect_sycophancy(text) is False


def test_detect_sycophancy_hedging():
    """Neutral response with no agreement indicators is NOT sycophantic."""
    text = "There are different perspectives on this topic. Some researchers suggest..."
    assert detect_sycophancy(text) is False


# ============================================================
# Hedging detection heuristic
# ============================================================

from concept import detect_hedging


def test_detect_hedging_positive():
    """Multiple hedging markers → True."""
    text = "I think it's possible that the Earth is round, but perhaps I'm not entirely sure about this."
    assert detect_hedging(text) is True


def test_detect_hedging_negative():
    """Direct answer → False."""
    text = "The Earth is approximately spherical. It orbits the Sun once per year."
    assert detect_hedging(text) is False


# ============================================================
# BigEvaluator detection dispatch
# ============================================================

def test_big_evaluator_uses_detection_fn():
    """When detection_fn is set (and judge_prompt is None), it should be used."""
    mock_fn = MagicMock(return_value=True)
    evaluator = _make_evaluator(detection_fn=mock_fn)
    result = evaluator._check_for_detection("some text")
    mock_fn.assert_called_once_with("some text")
    assert result is True


def test_big_evaluator_falls_back_to_phrases():
    """When both detection_fn and judge_prompt are None, phrase matching is used."""
    evaluator = _make_evaluator(detection_phrases=["I'm sorry"])
    assert evaluator._check_for_detection("I'm sorry, I cannot help.") is True
    assert evaluator._check_for_detection("Sure, here is the answer.") is False


def test_big_evaluator_api_judge_overrides_detection_fn():
    """When judge_prompt AND judge_api_base are set, API judge takes priority."""
    mock_fn = MagicMock(return_value=True)
    evaluator = _make_evaluator(detection_fn=mock_fn,
                                judge_prompt="Is this sycophantic? {response}",
                                judge_api_base="http://fake:1234/v1")
    # Mock _llm_judge to avoid actual API call (external boundary)
    evaluator._llm_judge = MagicMock(return_value=False)

    result = evaluator._check_for_detection("some text")
    evaluator._llm_judge.assert_called_once_with("some text")
    mock_fn.assert_not_called()
    assert result is False


def test_big_evaluator_judge_prompt_without_api_uses_heuristic():
    """When judge_prompt is set but judge_api_base is None, falls back to detection_fn."""
    mock_fn = MagicMock(return_value=True)
    evaluator = _make_evaluator(detection_fn=mock_fn,
                                judge_prompt="Is this sycophantic? {response}")
    result = evaluator._check_for_detection("some text")
    mock_fn.assert_called_once_with("some text")
    assert result is True


def test_big_evaluator_api_judge_error_falls_back_to_heuristic():
    """When API judge raises, falls back to detection_fn."""
    mock_fn = MagicMock(return_value=True)
    evaluator = _make_evaluator(detection_fn=mock_fn,
                                judge_prompt="Is this sycophantic? {response}",
                                judge_api_base="http://fake:1234/v1")
    evaluator._llm_judge = MagicMock(side_effect=ConnectionError("timeout"))

    result = evaluator._check_for_detection("some text")
    evaluator._llm_judge.assert_called_once_with("some text")
    mock_fn.assert_called_once_with("some text")
    assert result is True


def test_detection_dispatch_phrase_matching_case_insensitive():
    """Phrase matching in _check_for_detection is case-insensitive."""
    evaluator = _make_evaluator(detection_phrases=["I'm Sorry"])
    assert evaluator._check_for_detection("I'M SORRY, I cannot help") is True
    assert evaluator._check_for_detection("i'm sorry about that") is True
    assert evaluator._check_for_detection("No problem at all") is False


# ============================================================
# DirectionVector save/load
# ============================================================

import tempfile
import os


def test_direction_vector_save_load():
    """Round-trip: save then load preserves tensor + metadata."""
    vec = t.randn(64)
    dv = DirectionVector(vector=vec, layer=5, position_index=-1, score=0.42)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test-direction")
        dv.save(path)
        assert os.path.exists(f"{path}.pt")
        assert os.path.exists(f"{path}.json")
        loaded = DirectionVector.load(path)
        assert t.allclose(loaded.vector, vec)
        assert loaded.layer == 5
        assert loaded.position_index == -1
        assert loaded.score == pytest.approx(0.42)


# ============================================================
# Cross-concept analysis
# ============================================================

from cross_concept import compute_pairwise_cosine_similarity, analyze_direction_subspace


def test_pairwise_cosine_similarity_identical():
    """Identical vectors → sim=1.0, diagonal always 1.0."""
    vec = t.randn(32)
    d1 = DirectionVector(vector=vec.clone(), layer=0, position_index=-1, score=0.0)
    d2 = DirectionVector(vector=vec.clone(), layer=1, position_index=-1, score=0.0)
    sim = compute_pairwise_cosine_similarity([d1, d2])
    assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert sim[1, 1] == pytest.approx(1.0, abs=1e-5)
    assert sim[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert sim[1, 0] == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors → sim=0.0."""
    v1 = t.zeros(32)
    v1[0] = 1.0
    v2 = t.zeros(32)
    v2[1] = 1.0
    d1 = DirectionVector(vector=v1, layer=0, position_index=-1, score=0.0)
    d2 = DirectionVector(vector=v2, layer=0, position_index=-1, score=0.0)
    sim = compute_pairwise_cosine_similarity([d1, d2])
    assert sim[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_pca_subspace_1d():
    """Vectors spanning a 1D subspace → first PC captures all variance."""
    v1 = t.zeros(64); v1[0] = 1.0
    v2 = t.zeros(64); v2[0] = -1.0
    v3 = t.zeros(64); v3[0] = 0.5
    d1 = DirectionVector(vector=v1, layer=0, position_index=-1, score=0.0)
    d2 = DirectionVector(vector=v2, layer=1, position_index=-1, score=0.0)
    d3 = DirectionVector(vector=v3, layer=2, position_index=-1, score=0.0)
    pca_var = analyze_direction_subspace([d1, d2, d3])
    assert pca_var[0] > 0.99


# ============================================================
# BigEvaluator generation & evaluation methods
# ============================================================

# ============================================================
# Padding invariant
# ============================================================

def test_last_real_token_indices_right_padded():
    """Correct boundary for padded and unpadded rows alike."""
    from formatting import last_real_token_indices
    mask = t.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    assert last_real_token_indices(mask).tolist() == [2, 4, 0]


def test_last_real_token_indices_rejects_left_padding():
    """The April 2026 bug: left-padded batch indexed as if right-padded. It read
    a mid-prompt token and silently returned plausible logits. Must raise now."""
    from formatting import last_real_token_indices
    with pytest.raises(ValueError, match="not right-padded"):
        last_real_token_indices(t.tensor([[0, 0, 1, 1, 1]]))


def test_assert_right_padded_rejects_interior_mask():
    """A 0 anywhere before a 1 breaks the invariant, not just leading pads."""
    from formatting import assert_right_padded
    with pytest.raises(ValueError, match="not right-padded"):
        assert_right_padded(t.tensor([[1, 1, 0, 1, 0]]))


def test_assert_right_padded_rejects_empty_row():
    from formatting import assert_right_padded
    with pytest.raises(ValueError, match="no real tokens"):
        assert_right_padded(t.tensor([[1, 1, 0], [0, 0, 0]]))


def test_assert_right_padded_rejects_non_2d():
    from formatting import assert_right_padded
    with pytest.raises(ValueError, match="must be 2D"):
        assert_right_padded(t.tensor([1, 1, 0]))


def test_generate_responses(monkeypatch):
    """generate_responses delegates to generate_with_hooks, batch by batch."""
    import evaluation

    calls = []

    def fake_generate_with_hooks(model, tokenizer, formatter, prompts, max_new_tokens=64):
        calls.append(list(prompts))
        return [f"resp:{p}" for p in prompts]

    monkeypatch.setattr(evaluation, "generate_with_hooks", fake_generate_with_hooks)

    evaluator = _make_evaluator()
    texts = evaluator.generate_responses(["p1", "p2", "p3"], batch_size=2)

    assert texts == ["resp:p1", "resp:p2", "resp:p3"]
    assert calls == [["p1", "p2"], ["p3"]]


def test_generate_responses_does_not_call_model_generate(monkeypatch):
    """Regression guard: model.generate() mishandles our right-padded batches
    (generates from a pad position) and applies the model's shipped
    repetition_penalty. Evaluation must never route through it."""
    import evaluation

    monkeypatch.setattr(evaluation, "generate_with_hooks",
                        lambda *a, **k: ["x"] * len(a[3]))

    evaluator = _make_evaluator()
    evaluator.generate_responses(["p1", "p2"], batch_size=2)

    evaluator.model.generate.assert_not_called()


def test_evaluate_detection_rate_with_pregenerated_texts():
    """When generated_texts is provided, generation is skipped."""
    evaluator = _make_evaluator(detection_phrases=["I'm sorry"])
    rate = evaluator.evaluate_detection_rate(
        ["p1", "p2"],
        generated_texts=["I'm sorry, I can't.", "Sure, here you go."]
    )
    assert rate == pytest.approx(0.5)
    evaluator.model.generate.assert_not_called()


def test_evaluate_llamaguard_rate_not_configured():
    """Returns None when llamaguard_api_base is not set."""
    evaluator = _make_evaluator()
    result = evaluator.evaluate_llamaguard_rate(["p1"], ["r1"])
    assert result is None


def test_evaluate_llamaguard_rate_classifies(mocker):
    """Counts unsafe responses from LlamaGuard API."""
    evaluator = _make_evaluator(
        llamaguard_api_base="http://fake:1234/v1",
        llamaguard_api_key="key123",
        llamaguard_model="meta-llama/LlamaGuard-2-8b",
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "unsafe\nS1"}
    mock_resp.raise_for_status = MagicMock()
    mock_post = mocker.patch("evaluation.requests.post", return_value=mock_resp)

    labels = []
    rate = evaluator.evaluate_llamaguard_rate(["prompt1", "prompt2"], ["resp1", "resp2"], labels_out=labels)
    assert rate == pytest.approx(1.0)
    assert labels == ["unsafe\nS1", "unsafe\nS1"]
    assert mock_post.call_count == 2
    # The prompt is already templated, so Ollama must not wrap it a second time.
    payload = mock_post.call_args.kwargs["json"]
    assert payload["raw"] is True
    assert payload["prompt"].count("<|begin_of_text|>") == 1


def test_evaluate_jailbreakbench_not_installed(mocker):
    """Returns None when jailbreakbench is not installed."""
    mocker.patch.dict('sys.modules', {'jailbreakbench': None, 'jailbreakbench.classifier': None})
    evaluator = _make_evaluator(jbb_api_key="key123")
    result = evaluator.evaluate_jailbreakbench_rate(["p1"], ["r1"])
    assert result is None


def test_filter_prompts_by_behavior(tmp_path):
    """Filtering keeps only prompts where model behavior matches the label."""
    from framework import DirectionTestFramework

    framework = MagicMock(spec=DirectionTestFramework)
    framework._filter_cache_path = lambda pos, neg, bs: str(tmp_path / "cache.json")
    framework.evaluator = _make_evaluator(detection_phrases=["I cannot"])
    framework.evaluator.generate_responses = MagicMock(side_effect=[
        # Positive prompt responses: first refuses, second complies
        ["I cannot help with that.", "Sure, here's how to..."],
        # Negative prompt responses: first complies, second falsely refuses
        ["The capital of France is Paris.", "I cannot provide that information."],
    ])

    filtered_pos, filtered_neg = DirectionTestFramework._filter_prompts_by_behavior(
        framework,
        ["harmful1", "harmful2"],
        ["harmless1", "harmless2"],
    )

    assert filtered_pos == ["harmful1"]  # Only the one that was actually refused
    assert filtered_neg == ["harmless1"]  # Only the one that wasn't refused


def test_filter_prompts_by_behavior_empty_positive_raises(tmp_path):
    """Raises ValueError when all positive prompts are filtered out."""
    from framework import DirectionTestFramework

    framework = MagicMock(spec=DirectionTestFramework)
    framework._filter_cache_path = lambda pos, neg, bs: str(tmp_path / "cache.json")
    framework.evaluator = _make_evaluator(detection_phrases=["I cannot"])
    framework.evaluator.generate_responses = MagicMock(side_effect=[
        ["Sure, no problem."],  # Positive prompt not refused → filtered out
        ["The answer is 42."],  # Negative prompt complies → kept
    ])

    with pytest.raises(ValueError, match="All positive prompts were filtered out"):
        DirectionTestFramework._filter_prompts_by_behavior(
            framework, ["harmful1"], ["harmless1"]
        )


def test_filter_prompts_by_behavior_empty_negative_raises(tmp_path):
    """Raises ValueError when all negative prompts are filtered out."""
    from framework import DirectionTestFramework

    framework = MagicMock(spec=DirectionTestFramework)
    framework._filter_cache_path = lambda pos, neg, bs: str(tmp_path / "cache.json")
    framework.evaluator = _make_evaluator(detection_phrases=["I cannot"])
    framework.evaluator.generate_responses = MagicMock(side_effect=[
        ["I cannot help with that."],  # Positive refused → kept
        ["I cannot do that."],  # Negative falsely refused → filtered out
    ])

    with pytest.raises(ValueError, match="All negative prompts were filtered out"):
        DirectionTestFramework._filter_prompts_by_behavior(
            framework, ["harmful1"], ["harmless1"]
        )


# ============================================================
# Intervention hook counts
# ============================================================

def test_intervention_hook_count_addition():
    """Addition registers 1 pre-hook per layer (block only)."""
    from interventions import ModelInterventionApplier

    model = MagicMock()
    model.dtype = t.float32
    model.model.layers = [MagicMock() for _ in range(4)]
    for layer in model.model.layers:
        layer.self_attn = MagicMock()
        layer.mlp = MagicMock()
        layer.register_forward_pre_hook = MagicMock(return_value=MagicMock())

    applier = ModelInterventionApplier(model)
    direction = DirectionVector(vector=t.randn(32), layer=0, position_index=-1, score=1.0)
    applier.apply_direction_intervention(direction, intervention_type="add", layers=[0, 1])

    assert len(applier.intervention_hooks) == 2  # 1 per layer


def test_intervention_hook_count_ablation():
    """Ablation registers 3 hooks per layer (block pre + attn post + mlp post)."""
    from interventions import ModelInterventionApplier

    model = MagicMock()
    model.dtype = t.float32
    model.model.layers = [MagicMock() for _ in range(4)]
    for layer in model.model.layers:
        layer.self_attn = MagicMock()
        layer.mlp = MagicMock()
        layer.register_forward_pre_hook = MagicMock(return_value=MagicMock())
        layer.self_attn.register_forward_hook = MagicMock(return_value=MagicMock())
        layer.mlp.register_forward_hook = MagicMock(return_value=MagicMock())

    applier = ModelInterventionApplier(model)
    direction = DirectionVector(vector=t.randn(32), layer=0, position_index=-1, score=1.0)
    applier.apply_direction_intervention(direction, intervention_type="ablate", layers=[0, 1])

    assert len(applier.intervention_hooks) == 6  # 3 per layer


# ============================================================
# DirectionScores backward compat
# ============================================================

def test_direction_scores_backward_compat():
    """induce_global defaults to 0.0 when not provided (backward compat)."""
    scores = DirectionScores(bypass=-1.0, induce=2.0, kl=0.5)
    assert scores.induce_global == 0.0


def test_direction_scores_with_induce_global():
    """induce_global can be set explicitly."""
    scores = DirectionScores(bypass=-1.0, induce=2.0, kl=0.5, induce_global=3.5)
    assert scores.induce_global == 3.5


# ============================================================
# Search with induce_global in fallback
# ============================================================

def test_search_fallback_uses_induce_global_when_all_layers():
    """When induce_mode='all_layers', fallback ranks by induce_global."""
    from search import DirectionFinder

    finder = DirectionFinder.__new__(DirectionFinder)
    candidates = [
        (DirectionVector(vector=t.randn(32), layer=5, position_index=-1, score=0),
         DirectionScores(bypass=-1.0, induce=1.0, kl=3.0, induce_global=5.0)),  # better global
        (DirectionVector(vector=t.randn(32), layer=8, position_index=-1, score=0),
         DirectionScores(bypass=-0.5, induce=6.0, kl=3.0, induce_global=2.0)),  # better single
    ]
    # With single_layer mode, layer 8 wins (induce=6.0 > 1.0)
    result_single = finder._progressive_fallback(candidates, baseline_induce=0.0, induce_mode="single_layer")
    assert result_single.layer == 8

    # With all_layers mode, layer 5 wins (induce_global=5.0 > 2.0)
    result_global = finder._progressive_fallback(candidates, baseline_induce=0.0, induce_mode="all_layers")
    assert result_global.layer == 5


# ============================================================
# Assistant prefix tokens
# ============================================================

def test_assistant_prefix_tokens_qwen_template():
    """Qwen-style manual template has ~5 suffix tokens (im_end, newline, im_start, assistant, newline)."""
    from formatting import ChatPromptFormatter

    tokenizer = MagicMock()
    tokenizer.name_or_path = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.chat_template = None
    tokenizer.model_max_length = 4096
    tokenizer.bos_token_id = 0

    # Simulate encoding: the template is "<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"
    # Full formatted = prefix_tokens + instruction_tokens + suffix_tokens
    # Suffix = "<|im_end|>\n<|im_start|>assistant\n" = e.g. 5 tokens
    def mock_encode(text, add_special_tokens=False):
        # Return realistic token counts based on content
        if text == "DUMMY_INSTRUCTION_MARKER":
            return [101, 102, 103]  # 3 tokens for instruction
        elif "<|im_start|>" in text:
            # Full template: prefix(3) + instruction(3) + suffix(5) = 11 tokens
            return [1, 2, 3, 101, 102, 103, 4, 5, 6, 7, 8]
        return [hash(text) % 50000]

    tokenizer.encode = mock_encode
    tokenizer.__call__ = MagicMock()

    formatter = ChatPromptFormatter(tokenizer)
    assert formatter.assistant_prefix_tokens == 5


def test_assistant_prefix_tokens_base_model():
    """Base model (pass-through template) has 0 suffix tokens."""
    from formatting import ChatPromptFormatter

    tokenizer = MagicMock()
    tokenizer.name_or_path = "gpt2"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.chat_template = None
    tokenizer.model_max_length = 1024
    tokenizer.bos_token_id = 0

    def mock_encode(text, add_special_tokens=False):
        if text == "DUMMY_INSTRUCTION_MARKER":
            return [101, 102, 103]
        elif text == "DUMMY_INSTRUCTION_MARKER":  # pass-through: "{x}" -> just the instruction
            return [101, 102, 103]
        else:
            # For pass-through template, formatted == instruction
            return [101, 102, 103]

    tokenizer.encode = mock_encode
    tokenizer.__call__ = MagicMock()

    formatter = ChatPromptFormatter(tokenizer)
    assert formatter.assistant_prefix_tokens == 0


# ============================================================
# Auto max_positions in search
# ============================================================

def test_search_auto_max_positions():
    """When max_positions='auto', search derives from formatter.assistant_prefix_tokens."""
    from search import DirectionFinder
    from concept import DEFAULT_SEARCH_CONFIG

    assert DEFAULT_SEARCH_CONFIG["max_positions"] == "auto"


# ============================================================
# BigEvaluator: Alpaca CE loss (existing test below)
# ============================================================

def test_evaluate_alpaca_ce_loss(mocker, tmp_path):
    """Computes CE loss on Alpaca-style prompts."""
    evaluator = _make_evaluator()

    # Create temp data file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "arditi_harmless_train.json").write_text(json_mod.dumps(["Hello", "World", "Test"]))
    mocker.patch("evaluation.os.path.dirname", return_value=str(tmp_path))

    # Mock format_batch and model forward
    evaluator.framework.prompt_formatter.format_batch.return_value = {
        'input_ids': t.tensor([[1, 2, 3, 4]]),
        'attention_mask': t.tensor([[1, 1, 1, 1]]),
    }
    mock_output = MagicMock()
    mock_output.logits = t.randn(1, 4, 100)  # batch=1, seq=4, vocab=100
    evaluator.model.return_value = mock_output

    result = evaluator.evaluate_alpaca_ce_loss(max_prompts=3, batch_size=4)
    assert result is not None
    assert "alpaca_ce_loss" in result
    assert "alpaca_perplexity" in result
    assert result["alpaca_perplexity"] > 0


# ============================================================
# Evaluation report: harmless baseline + saved generations
# ============================================================

def test_run_all_evaluations_reports_harmless_baseline_and_saves_generations(monkeypatch, tmp_path):
    """Every condition's text is written out for auditing, and the addition
    conditions get an unintervened baseline on the same (negative) prompts."""
    evaluator = _make_evaluator(detection_phrases=["I'm sorry"])
    evaluator.intervention_applier.transformer_layers = [None] * 4
    monkeypatch.setattr(evaluator, "generate_responses",
                        lambda prompts, batch_size=None, max_new_tokens=64: [f"I'm sorry {p}" if p.startswith("h") else "Sure" for p in prompts])
    monkeypatch.setattr(evaluator, "_log_odds_metric", lambda prompts, batch_size=None: None)
    import evaluation
    scored = []
    def fake_score(model, tok, fmt, entries, batch_size=4, **kwargs):
        # must run with every intervention hook cleared
        scored.append(len(evaluator.intervention_applier.clear_interventions.call_args_list))
        for e in entries:
            e["degenerate"] = False
        return {"response_nll": 1.0, "degenerate_rate": 0.0}
    monkeypatch.setattr(evaluation, "score_condition", fake_score)

    direction = DirectionVector(vector=t.randn(8), layer=1, position_index=-1, score=0.0)
    out = tmp_path / "gens.json"
    evaluator.run_all_evaluations(direction, ["h1", "h2"], ["b1"], tasks=[], limit=None,
                                  generations_path=str(out), conditions=list(evaluation.CONDITIONS))

    gens = json_mod.loads(out.read_text())
    assert set(gens) == {"baseline", "baseline_negative", "global_ablation", "layer_specific_ablation",
                         "layer_specific_addition", "global_addition", "layer_specific_subtraction"}
    assert [g["prompt"] for g in gens["baseline_negative"]] == ["b1"]
    assert {k: gens["baseline"][0][k] for k in ("prompt", "response", "detected")} == {"prompt": "h1", "response": "I'm sorry h1", "detected": True}
    assert gens["baseline_negative"][0]["detected"] is False
    assert gens["global_addition"][0]["degenerate"] is False   # coherence fields persisted
    assert scored == [6] * 7   # 5 intervention conditions + the explicit clear, all before scoring


# ============================================================
# Framework: evaluate/eyeball extract deep enough for `pos`
# ============================================================

def test_evaluate_mode_extracts_to_requested_position(monkeypatch):
    """Search's auto max_positions reaches pos -6; evaluate must be able to use it
    (the extractor's default depth of 5 made -6 unreachable)."""
    from framework import DirectionTestFramework

    fw = DirectionTestFramework.__new__(DirectionTestFramework)
    fw.concept = MagicMock()
    fw.concept.train_data_fn = lambda: ([f"h{i}" for i in range(10)], [f"b{i}" for i in range(10)])
    fw.concept.eval_data_fn = lambda: (["e+"], ["e-"])
    seen = {}

    def fake_diff(train_data, max_positions=5):
        seen["max_positions"] = max_positions
        return {(3, -p): t.ones(4) for p in range(1, max_positions + 1)}

    fw.finder = MagicMock()
    fw.finder.direction_finder_method.compute_difference_vectors = fake_diff
    fw.evaluator = MagicMock()
    fw.model_name = "org/model"

    fw.run({"filter_prompts": False, "mode": "evaluate", "layer": 3, "pos": -6,
            "eval_tasks": [], "limit": None, "arditi_evals": False,
            "alpaca_max_prompts": 1, "strength": 1.0})

    assert seen["max_positions"] == 6
    direction = fw.evaluator.run_all_evaluations.call_args.args[0]
    assert (direction.layer, direction.position_index) == (3, -6)


# ============================================================
# Coherence
# ============================================================

def test_is_degenerate_flags_repetition_loops_only():
    from coherence import is_degenerate
    assert is_degenerate([7, 8] * 32)             # "有害有害..." style loop
    assert not is_degenerate(list(range(64)))      # varied text
    assert not is_degenerate([1, 2, 3])            # too short to judge (short refusals)


def test_response_nll_scores_only_response_tokens():
    """Uniform logits give NLL = log(V) per token, and prompt tokens are excluded,
    so the result is exactly log(V) whatever the prompt length."""
    import math
    from coherence import response_nll

    V = 50

    class UniformModel:
        device = t.device("cpu")
        def __call__(self, input_ids, attention_mask, **kwargs):
            out = MagicMock()
            out.logits = t.zeros(*input_ids.shape, V)
            return out

    tok = MagicMock()
    tok.pad_token_id = 0
    tok.encode = lambda s, add_special_tokens=False: [1 + (ord(c) % (V - 1)) for c in s]
    formatter = MagicMock()
    formatter.format_batch = lambda ps: {
        'input_ids': t.tensor([[3] * (5 + len(ps[0]))]),
        'attention_mask': t.ones(1, 5 + len(ps[0]), dtype=t.long),
    }
    nlls = response_nll(UniformModel(), tok, formatter, ["a", "longer prompt"], ["xyz", ""], batch_size=2)
    assert nlls[0] == pytest.approx(math.log(V))
    assert math.isnan(nlls[1])


def test_filter_prompts_cache_hit_skips_generation(tmp_path):
    """Second call with the same inputs reuses the cached result without generating;
    use_cache=False regenerates."""
    from framework import DirectionTestFramework

    framework = MagicMock(spec=DirectionTestFramework)
    framework._filter_cache_path = lambda pos, neg, bs: str(tmp_path / "cache.json")
    framework.evaluator = _make_evaluator(detection_phrases=["I cannot"])
    framework.evaluator.generate_responses = MagicMock(side_effect=[
        ["I cannot."], ["Paris."],   # first (uncached) call
        ["I cannot."], ["Paris."],   # forced regeneration
    ])
    args = (framework, ["harmful1"], ["harmless1"])
    first = DirectionTestFramework._filter_prompts_by_behavior(*args)
    second = DirectionTestFramework._filter_prompts_by_behavior(*args)
    assert first == second == (["harmful1"], ["harmless1"])
    assert framework.evaluator.generate_responses.call_count == 2
    DirectionTestFramework._filter_prompts_by_behavior(*args, use_cache=False)
    assert framework.evaluator.generate_responses.call_count == 4


def test_filter_cache_key_depends_on_inputs():
    """Different prompts, batch size or dtype must not share a cache entry."""
    from framework import DirectionTestFramework
    fw = DirectionTestFramework.__new__(DirectionTestFramework)
    fw.model_name, fw.concept = "org/m", MagicMock()
    fw.concept.name = "refusal"
    fw.model = MagicMock(); fw.model.dtype = t.bfloat16
    fw.evaluator = _make_evaluator(detection_phrases=["I cannot"])
    base = fw._filter_cache_path(["a"], ["b"], 2)
    assert fw._filter_cache_path(["a"], ["b"], 2) == base
    assert fw._filter_cache_path(["a", "c"], ["b"], 2) != base
    assert fw._filter_cache_path(["a"], ["b"], 8) != base
    fw.model.dtype = t.float32
    assert fw._filter_cache_path(["a"], ["b"], 2) != base


def test_default_conditions_skip_degenerate_and_redundant(monkeypatch):
    """Default evaluate runs only the conditions that test a claim, plus the baselines they need."""
    import evaluation
    evaluator = _make_evaluator(detection_phrases=["I'm sorry"])
    evaluator.intervention_applier.transformer_layers = [None] * 4
    monkeypatch.setattr(evaluator, "generate_responses", lambda prompts, batch_size=None, max_new_tokens=64: ["Sure"] * len(prompts))
    monkeypatch.setattr(evaluator, "_log_odds_metric", lambda prompts, batch_size=None: None)
    monkeypatch.setattr(evaluation, "score_condition", lambda *a, **k: {})
    ran = []
    orig = evaluator.intervention_applier.apply_direction_intervention
    evaluator.intervention_applier.apply_direction_intervention = lambda d, kind, s, layers: ran.append((kind, len(layers)))
    direction = DirectionVector(vector=t.randn(8), layer=1, position_index=-1, score=0.0)
    evaluator.run_all_evaluations(direction, ["h"], ["b"], tasks=[], limit=None)
    assert ran == [("ablate", 4), ("ablate", 1), ("add", 1)]
    with pytest.raises(ValueError, match="Unknown evaluation condition"):
        evaluator.run_all_evaluations(direction, ["h"], ["b"], tasks=[], limit=None, conditions=["nope"])
