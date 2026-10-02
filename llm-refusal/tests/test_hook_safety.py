"""Tests that hook registration/removal survives exceptions (see CLAUDE.md's
"Hook cleanup" gotcha: leaked hooks corrupt results in later calls).

Covers:
- activations.py: extract_residual_activations must remove pre-hooks even if
  the model's forward pass raises mid-extraction.
- scoring.py: _get_logits must clear intervention hooks even if the model's
  forward pass raises mid-scoring (after intervention hooks were applied).
- interventions.py: apply_direction_intervention must leave zero hooks behind
  if registration fails partway through the per-layer loop.
"""
import torch as t
import pytest
from unittest.mock import MagicMock

from activations import ActivationExtractor
from interventions import ModelInterventionApplier
from scoring import Three_Score_Evaluator
from datatypes import DirectionVector
from tests.fakes import ResidualModel, SimpleTokenizer, no_hooks


HIDDEN = 8


# ============================================================
# activations.py: extract_residual_activations hook cleanup
# ============================================================

def test_extract_residual_activations_removes_hooks_on_forward_failure():
    """Pre-hooks must be removed even when the model's forward pass raises
    mid-extraction."""
    model = ResidualModel(HIDDEN, n_layers=3, fail_at_layer=1)
    layers = model.model.layers

    formatter = MagicMock()
    formatter.format_batch.return_value = {
        'input_ids': t.ones(2, 4, dtype=t.long),
        'attention_mask': t.ones(2, 4, dtype=t.long),
        'position_ids': t.zeros(2, 4, dtype=t.long),
    }

    extractor = ActivationExtractor(model, tokenizer=MagicMock(), transformer_layers=layers,
                                     prompt_formatter=formatter)

    with pytest.raises(RuntimeError, match="simulated forward failure"):
        extractor.extract_residual_activations(["p1", "p2"], max_positions=3)

    for layer in layers:
        assert len(layer._forward_pre_hooks) == 0, "pre-hook leaked after forward failure"


# ============================================================
# scoring.py: _get_logits hook cleanup
# ============================================================

def test_get_logits_clears_intervention_hooks_on_forward_failure():
    """apply_direction_intervention() registers real hooks, then the model's
    forward pass raises — the finally block must still clear them."""
    model = ResidualModel(HIDDEN, n_layers=3, fail_at_layer=0)
    layers = model.model.layers
    applier = ModelInterventionApplier(model)

    formatter = MagicMock()
    formatter.format_batch.return_value = {
        'input_ids': t.ones(1, 4, dtype=t.long),
        'attention_mask': t.ones(1, 4, dtype=t.long),
        'position_ids': t.zeros(1, 4, dtype=t.long),
    }

    evaluator = Three_Score_Evaluator(model, SimpleTokenizer({"yes": 1, "no": 2}), applier, formatter,
                                       target_tokens=["yes", "no"])

    direction = DirectionVector(vector=t.randn(HIDDEN), layer=0, position_index=-1, score=0.0)

    with pytest.raises(RuntimeError, match="simulated forward failure"):
        evaluator._get_logits(["p1"], intervention=(direction, "ablate", [0, 1, 2]))

    assert applier.intervention_hooks == []
    assert no_hooks(layers)


# ============================================================
# interventions.py: atomic registration
# ============================================================

def test_apply_direction_intervention_rolls_back_on_partial_registration_failure():
    """If _get_sublayers raises partway through the per-layer loop (e.g. an
    unknown architecture), zero hooks remain registered. The rollback is
    clear_interventions(), so hooks from an earlier successful call go too."""
    model = ResidualModel(HIDDEN, n_layers=3)
    layers = model.model.layers
    applier = ModelInterventionApplier(model)
    direction = DirectionVector(vector=t.randn(HIDDEN), layer=0, position_index=-1, score=0.0)
    applier.apply_direction_intervention(direction, intervention_type="add", layers=[0])
    assert len(applier.intervention_hooks) == 1

    original_get_sublayers = applier._get_sublayers
    call_count = {"n": 0}

    def flaky_get_sublayers(block):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise AttributeError("simulated unknown architecture")
        return original_get_sublayers(block)

    applier._get_sublayers = flaky_get_sublayers

    with pytest.raises(AttributeError, match="simulated unknown architecture"):
        applier.apply_direction_intervention(direction, intervention_type="ablate", layers=[0, 1, 2])

    assert applier.intervention_hooks == []
    assert no_hooks(layers)


def test_intervened_clears_hooks_on_normal_exit_and_exception():
    """`with applier.intervened(...)` registers hooks inside the block and removes
    them on exit, including when the body raises."""
    model = ResidualModel(HIDDEN, n_layers=3)
    layers = model.model.layers
    applier = ModelInterventionApplier(model)
    direction = DirectionVector(vector=t.randn(HIDDEN), layer=0, position_index=-1, score=0.0)

    with applier.intervened(direction, "ablate", layers=[0, 1, 2]):
        assert len(applier.intervention_hooks) == 9   # block pre + attn/mlp post per layer
    assert applier.intervention_hooks == []
    assert no_hooks(layers)

    with pytest.raises(RuntimeError, match="boom"):
        with applier.intervened(direction, "add", 1.0, layers=[1]):
            assert len(applier.intervention_hooks) == 1
            raise RuntimeError("boom")
    assert applier.intervention_hooks == []
    assert no_hooks(layers)


def test_ablating_zero_norm_direction_raises_and_leaves_no_hooks():
    """A zero direction would normalise to NaN and poison every hidden state, so
    ablation must refuse it up front. 'add' of a zero vector is a harmless no-op."""
    model = ResidualModel(HIDDEN, n_layers=2)
    layers = model.model.layers
    applier = ModelInterventionApplier(model)
    zero = DirectionVector(vector=t.zeros(HIDDEN), layer=0, position_index=-1, score=0.0)

    with pytest.raises(ValueError, match="norm"):
        applier.apply_direction_intervention(zero, "ablate")
    assert applier.intervention_hooks == []
    assert no_hooks(layers)

    with applier.intervened(zero, "add", 1.0, layers=[0]):
        assert len(applier.intervention_hooks) == 1
    assert no_hooks(layers)
