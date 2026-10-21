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


def test_none_subset_pushes_generated_tokens_only():
    pp = _mod()
    m = _Model()
    with pp.positional_push(m, m.block, t.tensor([1.0, 2.0]), "none", n_suffix=2, generated=True):
        m(input_ids=t.zeros(2, 4, dtype=t.long), attention_mask=t.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]))
        m(input_ids=t.zeros(2, 1, dtype=t.long), attention_mask=t.ones(2, 5))
    pre, step = m.seen
    assert pre.abs().sum() == 0
    assert step[..., 1].tolist() == [[2.0], [2.0]]


def test_positional_push_first_k_generated_tokens_resets_at_each_prefill():
    pp = _mod()
    am = t.ones(1, 3, dtype=t.long)
    v = t.tensor([1.0, 0.0])
    m = _Model()
    with pp.positional_push(m, m.block, v, "last", n_suffix=1, generated=2):
        for _ in range(2):                                                     # two generations (batches)
            m(input_ids=t.zeros(1, 3, dtype=t.long), attention_mask=am)
            for _ in range(4):
                m(input_ids=t.zeros(1, 1, dtype=t.long), attention_mask=am)
    steps = [x[0, 0, 0].item() for x in m.seen if x.shape[1] == 1]
    assert steps == [1, 1, 0, 0] * 2                                            # only decode steps 1-2 pushed
    m2 = _Model()
    with pp.positional_push(m2, m2.block, v, "last", n_suffix=1, generated=True):  # True is "all", not 1
        m2(input_ids=t.zeros(1, 3, dtype=t.long), attention_mask=am)
        for _ in range(3):
            m2(input_ids=t.zeros(1, 1, dtype=t.long), attention_mask=am)
    assert [x[0, 0, 0].item() for x in m2.seen if x.shape[1] == 1] == [1, 1, 1]


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
    js.positional_push = lambda model, block, vec, subset, n, generated, n_header=0: (
        calls.append((vec.tolist(), subset, n, generated)) or contextlib.nullcontext())

    class Fmt:
        assistant_prefix_tokens = 5

    class Run:
        model, blocks, fmt = None, [None], Fmt()
    with js.condition_context(Run, f"subpos:template+gen:{stem}*0.5"):
        pass
    with js.condition_context(Run, f"addpos:instruction:{stem}"):
        pass
    with js.condition_context(Run, f"addpos:all+gen16:{stem}"):
        pass
    assert calls == [([-0.5, 0.0], "template", 5, True), ([1.0, 0.0], "instruction", 5, False),
                     ([1.0, 0.0], "all", 5, 16)]
    import pytest
    with pytest.raises(AssertionError):
        js.condition_context(Run, f"addpos:all+foo:{stem}")


def test_jbb_safety_eval_prompts_switches_to_the_harmless_set():
    spec = importlib.util.spec_from_file_location(
        "jbb_safety", os.path.join(os.path.dirname(__file__), "..", "scripts", "jbb_safety.py"))
    js = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(js)
    harmful, harmless = js.eval_prompts(False), js.eval_prompts(True)
    assert len(harmful) == len(harmless) == 100 and not set(harmful) & set(harmless)


def test_generated_ablation_touches_decode_steps_only():
    pp = _mod()
    m = _Model()
    m.block.register_forward_pre_hook(lambda mod, args: (args[0] + t.tensor([3.0, 4.0]),))  # a component to remove
    u = t.tensor([1.0, 0.0])
    with pp.generated_ablation([m.block], u):
        m(input_ids=t.zeros(1, 3, dtype=t.long), attention_mask=t.ones(1, 3))     # prefill: untouched
        m(input_ids=t.zeros(1, 1, dtype=t.long), attention_mask=t.ones(1, 4))     # decode: u removed
    pre, step = m.seen
    assert pre[0, :, 0].tolist() == [3.0, 3.0, 3.0] and step[0, 0].tolist() == [0.0, 4.0]


def test_header_and_request_split_the_instruction():
    pp = _mod()
    am = t.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
    m = pp.position_masks(am, n_suffix=2, n_header=2)
    assert m["header"].int().tolist() == [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]
    assert m["request"].int().tolist() == [[0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0]]
    assert (m["header"] | m["request"]).equal(m["instruction"])


def test_header_tokens_is_the_shared_prefix_before_the_users_text():
    pp = _mod()

    class Fmt:
        def format_batch(self, prompts):
            ids = [1, 50, 51] + [ord(c) for c in prompts[0]] + [60, 61]   # BOS, header, text, suffix
            return {"input_ids": t.tensor([ids])}
    assert pp.header_tokens(Fmt()) == 3
