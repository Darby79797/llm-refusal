"""Tests for evaluate_llamaguard_rate's denominator handling (FIX 3).

Per-item API failures were previously logged and skipped but still counted
against len(prompts) in the denominator, silently deflating the reported
unsafe rate (e.g. 5 timeouts out of 100 reported as 95/100 instead of 95/95).
"""
import pytest
import torch as t
from unittest.mock import MagicMock

from evaluation import BigEvaluator


def _make_evaluator(**kwargs):
    framework = MagicMock()
    framework.model.device = t.device("cpu")
    framework.intervention_applier.transformer_layers = []
    return BigEvaluator(framework, **kwargs)


def test_evaluate_llamaguard_rate_excludes_failures_from_denominator(mocker):
    """Rate = unsafe_count / num_classified, not len(prompts)."""
    evaluator = _make_evaluator(
        llamaguard_api_base="http://fake:1234/v1",
        llamaguard_model="llama-guard2",
    )

    responses = [
        {"response": "unsafe\nS1"},
        {"response": "safe"},
        None,  # will raise
        {"response": "unsafe\nS9"},
        {"response": "safe"},
    ]

    def fake_post(url, json=None, timeout=None):
        item = responses.pop(0)
        if item is None:
            raise ConnectionError("simulated timeout")
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = item
        return resp

    mocker.patch("evaluation.requests.post", side_effect=fake_post)

    prompts = [f"p{i}" for i in range(5)]
    texts = [f"r{i}" for i in range(5)]
    rate = evaluator.evaluate_llamaguard_rate(prompts, texts)

    # 2 unsafe out of 4 successfully classified (1 failed and is excluded).
    assert rate == pytest.approx(0.5)


def test_evaluate_llamaguard_rate_all_failures_returns_none(mocker):
    """When every classification call fails, return None (matches
    evaluate_jailbreakbench_rate's whole-batch-failure semantics)."""
    evaluator = _make_evaluator(llamaguard_api_base="http://fake:1234/v1")

    mocker.patch("evaluation.requests.post", side_effect=ConnectionError("down"))

    rate = evaluator.evaluate_llamaguard_rate(["p1", "p2", "p3"], ["r1", "r2", "r3"])
    assert rate is None


def test_evaluate_llamaguard_rate_no_failures_unchanged(mocker):
    """No failures: behaves exactly as before (denominator == len(prompts))."""
    evaluator = _make_evaluator(llamaguard_api_base="http://fake:1234/v1")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "unsafe\nS1"}
    mock_resp.raise_for_status = MagicMock()
    mocker.patch("evaluation.requests.post", return_value=mock_resp)

    rate = evaluator.evaluate_llamaguard_rate(["prompt1", "prompt2"], ["resp1", "resp2"])
    assert rate == 1.0


def test_evaluate_llamaguard_rate_empty_prompts_returns_zero():
    """Empty input is not a failure — preserves the original 0.0-for-empty behavior."""
    evaluator = _make_evaluator(llamaguard_api_base="http://fake:1234/v1")
    assert evaluator.evaluate_llamaguard_rate([], []) == 0.0
