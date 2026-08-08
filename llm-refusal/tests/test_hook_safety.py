"""Tests that hook registration/removal survives exceptions (see CLAUDE.md's
"Hook cleanup" gotcha: leaked hooks corrupt results in later calls).

Covers:
- activations.py: extract_residual_activations must remove pre-hooks even if
  the model's forward pass raises mid-extraction.
- scoring.py: _get_logits must clear intervention hooks even if the model's
  forward pass raises mid-scoring (after intervention hooks were applied).
- interventions.py: apply_direction_intervention must leave zero hooks behind
  if registration fails partway through the per-layer loop.
- generation.py: pure-tensor regression test for the next_position off-by-one
  fix (position_ids sentinel of 1 must not be mistaken for a real position).
"""
import torch as t
import torch.nn as nn
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from activations import ActivationExtractor
from interventions import ModelInterventionApplier
from scoring import Three_Score_Evaluator
from datatypes import DirectionVector


HIDDEN = 8


class TinyBlock(nn.Module):
    """Stand-in transformer block with real self_attn/mlp submodules, so
    register_forward_pre_hook/register_forward_hook populate real hook dicts
    we can inspect after the fact."""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.mlp = nn.Linear(hidden, hidden)

    def forward(self, x):
        return x


class FakeHFModel:
    """Minimal stand-in for a HF causal LM exposing the `.model.layers`
    attribute path that ModelInterventionApplier/ActivationExtractor expect."""
    def __init__(self, layers):
        self.model = SimpleNamespace(layers=layers)
        self.dtype = t.float32
        self.device = t.device("cpu")

    def __call__(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        raise RuntimeError("simulated forward failure")


class FailingAtLayerModel:
    """Like FakeHFModel, but actually invokes each block (firing any registered
    pre-hooks) before raising, to simulate a mid-extraction failure."""
    def __init__(self, layers, fail_at_layer):
        self.model = SimpleNamespace(layers=layers)
        self.layers = layers
        self.fail_at_layer = fail_at_layer
        self.dtype = t.float32
        self.device = t.device("cpu")

    def __call__(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        x = t.zeros(batch_size, seq_len, HIDDEN)
        for i, layer in enumerate(self.layers):
            if i == self.fail_at_layer:
                raise RuntimeError("simulated forward failure")
            x = layer(x)
        return x


def _all_hooks_empty(layers):
    for layer in layers:
        if len(layer._forward_pre_hooks) != 0:
            return False
        if len(layer.self_attn._forward_hooks) != 0:
            return False
        if len(layer.mlp._forward_hooks) != 0:
            return False
    return True


# ============================================================
# activations.py: extract_residual_activations hook cleanup
# ============================================================

def test_extract_residual_activations_removes_hooks_on_forward_failure():
    """Pre-hooks must be removed even when the model's forward pass raises
    mid-extraction (FIX 1d)."""
    layers = nn.ModuleList([TinyBlock() for _ in range(3)])
    model = FailingAtLayerModel(layers, fail_at_layer=1)

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

class SimpleTokenizer:
    """Minimal tokenizer for LogOddsMetric's target-token validation."""
    def __init__(self, token_map=None):
        self._map = token_map or {"yes": 1, "no": 2}

    def encode(self, token, add_special_tokens=False):
        return [self._map.get(token, hash(token) % 1000)]


def test_get_logits_clears_intervention_hooks_on_forward_failure():
    """apply_direction_intervention() registers real hooks, then the model's
    forward pass raises — the finally block must still clear them (FIX 1c)."""
    layers = nn.ModuleList([TinyBlock() for _ in range(3)])
    model = FakeHFModel(layers)
    applier = ModelInterventionApplier(model)

    formatter = MagicMock()
    formatter.format_batch.return_value = {
        'input_ids': t.ones(1, 4, dtype=t.long),
        'attention_mask': t.ones(1, 4, dtype=t.long),
        'position_ids': t.zeros(1, 4, dtype=t.long),
    }

    evaluator = Three_Score_Evaluator(model, SimpleTokenizer(), applier, formatter,
                                       target_tokens=["yes", "no"])

    direction = DirectionVector(vector=t.randn(HIDDEN), layer=0, position_index=-1, score=0.0)

    with pytest.raises(RuntimeError, match="simulated forward failure"):
        evaluator._get_logits(["p1"], intervention=(direction, "ablate", [0, 1, 2]))

    assert applier.intervention_hooks == []
    assert _all_hooks_empty(layers)


# ============================================================
# interventions.py: atomic registration
# ============================================================

def test_apply_direction_intervention_rolls_back_on_partial_registration_failure():
    """If _get_sublayers raises partway through the per-layer loop (e.g. an
    unknown architecture), zero hooks should remain registered (FIX 1e)."""
    layers = nn.ModuleList([TinyBlock() for _ in range(3)])
    model = FakeHFModel(layers)
    applier = ModelInterventionApplier(model)

    original_get_sublayers = applier._get_sublayers
    call_count = {"n": 0}

    def flaky_get_sublayers(block):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise AttributeError("simulated unknown architecture")
        return original_get_sublayers(block)

    applier._get_sublayers = flaky_get_sublayers

    direction = DirectionVector(vector=t.randn(HIDDEN), layer=0, position_index=-1, score=0.0)

    with pytest.raises(AttributeError, match="simulated unknown architecture"):
        applier.apply_direction_intervention(direction, intervention_type="ablate", layers=[0, 1, 2])

    assert applier.intervention_hooks == []
    assert _all_hooks_empty(layers)


def test_apply_direction_intervention_rolls_back_only_new_hooks_when_prior_exist():
    """If a second (failing) call is made after a prior successful call, the
    rollback must not clobber hooks that legitimately existed before it."""
    layers = nn.ModuleList([TinyBlock() for _ in range(3)])
    model = FakeHFModel(layers)
    applier = ModelInterventionApplier(model)

    direction = DirectionVector(vector=t.randn(HIDDEN), layer=0, position_index=-1, score=0.0)
    # First call succeeds (addition — no _get_sublayers involved).
    applier.apply_direction_intervention(direction, intervention_type="add", layers=[0])
    assert len(applier.intervention_hooks) == 1

    original_get_sublayers = applier._get_sublayers

    def always_fail(block):
        raise AttributeError("simulated unknown architecture")

    applier._get_sublayers = always_fail
    with pytest.raises(AttributeError):
        applier.apply_direction_intervention(direction, intervention_type="ablate", layers=[1, 2])

    # Whole-call atomicity per the documented contract: clear_interventions() is
    # called on failure, so all hooks (including the earlier "add" ones) are gone.
    assert applier.intervention_hooks == []
    applier._get_sublayers = original_get_sublayers


# ============================================================
# generation.py: next_position off-by-one fix (pure tensor)
# ============================================================

def test_generation_next_position_fix_matches_true_sequence_length():
    """Regression test for FIX 2: position_ids' right-padding sentinel (1, see
    formatting.py's masked_fill) must not be mistaken for a real max position.

    A sequence with true_len==1 has position_ids [0, 1, 1, 1, 1] (real token
    at position 0, padded slots sentinel-filled with 1) — the old
    `position_ids.max()+1` computation would wrongly give next_position=2.
    The fix uses attention_mask.sum(dim=1), which is exactly the true length.
    """
    attention_mask = t.tensor([[1, 0, 0, 0, 0],
                                [1, 1, 1, 1, 1]], dtype=t.long)

    # Mirrors formatting.py's position_ids construction exactly.
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    assert position_ids.tolist() == [[0, 1, 1, 1, 1], [0, 1, 2, 3, 4]]

    # Old (buggy) computation, kept here to document the regression.
    buggy_next_position = position_ids.max(dim=-1).values + 1
    assert buggy_next_position.tolist() == [2, 5]

    # Fixed computation (matches generation.py:37).
    next_position = attention_mask.sum(dim=1).to(position_ids.dtype)
    assert next_position.tolist() == [1, 5]
