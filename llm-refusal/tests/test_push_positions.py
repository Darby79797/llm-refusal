"""push_positions.py: token-subset masks under right padding, and the share of the full push's effect."""
import importlib.util
import math
import os

import torch as t


def _mod():
    spec = importlib.util.spec_from_file_location(
        "push_positions", os.path.join(os.path.dirname(__file__), "..", "scripts", "push_positions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_position_masks_follow_each_rows_real_length():
    pp = _mod()
    am = t.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])     # right-padded: true lengths 6 and 4
    m = pp.position_masks(am, n_suffix=2)
    assert m["template"].tolist() == [[0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0]]
    assert m["instruction"].tolist() == [[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]]
    assert m["last"].tolist() == [[0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]]
    assert m["all"].tolist() == am.bool().tolist()
    assert not (m["template"] & m["instruction"]).any()
    assert ((m["template"] | m["instruction"]) == m["all"]).all()


def test_share_is_the_fraction_of_the_full_effect():
    pp = _mod()
    assert pp.share(-2.0, -4.0, 6.0) == 0.8
    assert pp.share(6.0, -4.0, 6.0) == 0.0
    assert math.isnan(pp.share(1.0, 6.2, 6.0))   # the full push barely moves: no share


class _Block(t.nn.Module):
    def forward(self, h):
        return h


class _Model(t.nn.Module):
    """A one-block stand-in that records what its block received (prefill [B, T, d], then one-token decode steps)."""
    def __init__(self):
        super().__init__()
        self.block, self.seen = _Block(), []

    def forward(self, input_ids=None, attention_mask=None):
        h = t.zeros(*input_ids.shape, 2)
        self.seen.append(self.block(h))
        return h


def test_positional_push_adds_on_the_subset_at_prefill_and_on_generated_tokens_only_with_gen():
    pp = _mod()
    am = t.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    v = t.tensor([1.0, 2.0])
    for gen in (False, True):
        m = _Model()
        with pp.positional_push(m, m.block, v, "template", n_suffix=2, generated=gen):
            m(input_ids=t.zeros(2, 4, dtype=t.long), attention_mask=am)          # prefill
            m(input_ids=t.zeros(2, 1, dtype=t.long), attention_mask=t.ones(2, 5))  # one decode step
        pre, step = m.seen
        assert pre[..., 0].tolist() == [[0, 0, 1, 1], [0, 1, 1, 0]]
        assert step[..., 1].tolist() == ([[2.0], [2.0]] if gen else [[0.0], [0.0]])
    m(input_ids=t.zeros(2, 4, dtype=t.long), attention_mask=am)                     # hooks removed on exit
    assert m.seen[-1].abs().sum() == 0


def test_jbb_condition_subpos_parses_subset_gen_and_strength(tmp_path):
    import contextlib
    spec = importlib.util.spec_from_file_location(
        "jbb_safety", os.path.join(os.path.dirname(__file__), "..", "scripts", "jbb_safety.py"))
    js = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(js)
    from datatypes import DirectionVector
    stem = str(tmp_path / "x-direction")
    DirectionVector(vector=t.tensor([1.0, 0.0]), layer=0, position_index=-1, score=0.0).save(stem)
    calls = []
    js.positional_push = lambda model, block, vec, subset, n, generated: (
        calls.append((vec.tolist(), subset, n, generated)) or contextlib.nullcontext())

    class Fmt:
        assistant_prefix_tokens = 5

    class Run:
        model, blocks, fmt = None, [None], Fmt()
    with js.condition_context(Run, f"subpos:template+gen:{stem}*0.5"):
        pass
    with js.condition_context(Run, f"addpos:instruction:{stem}"):
        pass
    assert calls == [([-0.5, 0.0], "template", 5, True), ([1.0, 0.0], "instruction", 5, False)]
