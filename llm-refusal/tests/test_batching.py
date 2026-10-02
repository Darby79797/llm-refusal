"""Batch-size resolution, OOM-safe batching, and batch-invariance of generation.

The invariance tests use a tiny randomly-initialised Qwen2 (2 layers, CPU) with
the real Qwen2.5 tokenizer and chat template, so they exercise the real padding,
KV-cache and hook paths in a few seconds. On CPU, generation is exactly
batch-invariant in fp32 and bf16 alike. The few-pp drift seen with bf16 on MPS
comes from its kernels, so these tests pin down what our code controls, not
MPS numerics.
"""
import pytest
import torch as t
from unittest.mock import MagicMock

import batching
from batching import (AUTO, estimate_bytes_per_row, map_batched, parse_batch_size,
                      resolve_batch_size)
from datatypes import DirectionVector
from tests.fakes import cached_tokenizer, tiny_qwen2

# Compared against a bs=1 reference, so 1 itself is left out (it would compare
# with itself). Powers of two, plus one that isn't.
BATCH_SIZES = [2, 3, 4, 8, 16]


# ---------------------------------------------------------------- parsing

@pytest.mark.parametrize("value,expected", [("auto", AUTO), ("AUTO", AUTO), ("8", 8), (3, 3)])
def test_parse_batch_size(value, expected):
    assert parse_batch_size(value) == expected


@pytest.mark.parametrize("bad", ["0", -1, "x"])
def test_parse_batch_size_rejects(bad):
    with pytest.raises(ValueError):
        parse_batch_size(bad)


# ---------------------------------------------------------------- estimation / resolution

def _config(**kw):
    c = MagicMock()
    c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads = 32, 32, 8
    c.head_dim, c.hidden_size, c.intermediate_size, c.vocab_size = 128, 4096, 14336, 128256
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_estimate_grows_with_prompt_and_generation_length():
    cfg = _config()
    base = estimate_bytes_per_row(cfg, t.bfloat16, 50, 64)
    assert estimate_bytes_per_row(cfg, t.bfloat16, 50, 512) > base
    assert estimate_bytes_per_row(cfg, t.bfloat16, 200, 64) > base
    assert estimate_bytes_per_row(cfg, t.float32, 50, 64) == 2 * base


def _fake_model():
    model = MagicMock()
    model.config = _config()
    model.dtype = t.bfloat16
    model.device = t.device("cpu")
    return model


def _formatter(tokens_per_prompt):
    f = MagicMock()
    f.format_batch = lambda ps: {'attention_mask': t.ones(1, tokens_per_prompt, dtype=t.long)}
    return f


def test_resolve_passes_explicit_int_through():
    assert resolve_batch_size(5, MagicMock(), MagicMock(), ["p"], 64) == 5


@pytest.mark.parametrize("budget_rows,expected", [(0.5, 1), (1, 1), (3, 2), (16, 16), (17, 16), (1000, 64)])
def test_resolve_auto_picks_largest_power_of_two_in_budget(monkeypatch, budget_rows, expected):
    per_row = 100_000_000
    monkeypatch.setattr(batching, "estimate_bytes_per_row", lambda *a, **k: per_row)
    monkeypatch.setattr(batching, "weight_bytes", lambda m: 0)
    monkeypatch.setattr(batching, "device_total_bytes", lambda d: int(budget_rows * per_row / batching.AUTO_MEMORY_FRACTION))
    got = resolve_batch_size(AUTO, _fake_model(), _formatter(10), ["a", "b"], 64)
    assert got == expected


def test_resolve_auto_is_deterministic_and_shrinks_with_length(monkeypatch):
    monkeypatch.setattr(batching, "weight_bytes", lambda m: 16 * 10**9)
    monkeypatch.setattr(batching, "device_total_bytes", lambda d: 38 * 10**9)
    model, fmt = _fake_model(), _formatter(60)
    short = resolve_batch_size(AUTO, model, fmt, ["p"] * 10, 64)
    assert short == resolve_batch_size(AUTO, model, fmt, ["p"] * 10, 64)
    assert resolve_batch_size(AUTO, model, fmt, ["p"] * 10, 512) <= short


# ---------------------------------------------------------------- map_batched

def _oom():
    return RuntimeError("MPS backend out of memory (MPS allocated: 30 GB)")


def test_map_batched_preserves_order_and_chunking():
    seen = []
    out = map_batched(lambda c: (seen.append(list(c)), [x * 10 for x in c])[1], list(range(7)), 3)
    assert out == [0, 10, 20, 30, 40, 50, 60]
    assert seen == [[0, 1, 2], [3, 4, 5], [6]]


def test_map_batched_splits_on_oom_and_reports():
    splits = []
    def fn(chunk):
        if len(chunk) > 2:
            raise _oom()
        return [x + 1 for x in chunk]
    out = map_batched(fn, list(range(8)), 8, on_split=splits.append)
    assert out == [x + 1 for x in range(8)]
    assert splits == [8, 4, 4]  # 8 -> 4+4 -> each 4 -> 2+2


def test_map_batched_reraises_single_item_oom_and_non_oom():
    with pytest.raises(RuntimeError, match="out of memory"):
        map_batched(lambda c: (_ for _ in ()).throw(_oom()), [1], 4)
    with pytest.raises(ValueError, match="boom"):
        map_batched(lambda c: (_ for _ in ()).throw(ValueError("boom")), [1, 2], 2)


def test_map_batched_rejects_wrong_result_count():
    with pytest.raises(ValueError, match="returned 1 results for 2 items"):
        map_batched(lambda c: [0], [1, 2], 2)


# ---------------------------------------------------------------- real (tiny) model

PROMPTS = [("word " * k).strip() + "?" for k in (1, 3, 7, 2, 15, 4, 9, 1, 30, 5, 11, 2, 6, 20, 8, 3)]


@pytest.fixture(scope="module")
def tiny_setup():
    """Tiny random Qwen2 + the real Qwen2.5 tokenizer/template, CPU."""
    from formatting import ChatPromptFormatter
    from interventions import ModelInterventionApplier
    tok = cached_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    model = tiny_qwen2(tok, hidden_size=64, tie_word_embeddings=True,
                       # The default 0.02 init makes the model emit "\n" forever for
                       # every prompt, which would make the invariance tests vacuous.
                       initializer_range=0.2)
    model.generation_config.eos_token_id = None  # random weights: never stop early
    fmt = ChatPromptFormatter(tok)
    framework = MagicMock()
    framework.model, framework.tokenizer, framework.prompt_formatter = model, tok, fmt
    framework.intervention_applier = ModelInterventionApplier(model)
    return framework


def _evaluator(framework, batch_size):
    from evaluation import BigEvaluator
    return BigEvaluator(framework, gen_batch_size=batch_size)


@pytest.fixture(scope="module")
def reference(tiny_setup):
    ev = _evaluator(tiny_setup, 1)
    ref = {"plain": ev.generate_responses(PROMPTS, max_new_tokens=12)}
    assert len(set(ref["plain"])) == len(set(PROMPTS)), "outputs don't depend on the prompt; test would be vacuous"
    tiny_setup.intervention_applier.apply_direction_intervention(
        DirectionVector(t.randn(64, generator=t.Generator().manual_seed(1)), 1, -1, 0), "ablate", layers=[0, 1])
    try:
        ref["ablate"] = ev.generate_responses(PROMPTS, max_new_tokens=12)
    finally:
        tiny_setup.intervention_applier.clear_interventions()
    assert ref["plain"] != ref["ablate"], "ablation hook had no effect; test would be vacuous"
    return ref


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_generation_is_batch_invariant(tiny_setup, reference, batch_size):
    """Every batch size, with and without ablation hooks, must give the
    bs=1 output token for token (mixed prompt lengths exercise the padding)."""
    ev = _evaluator(tiny_setup, batch_size)
    assert ev.generate_responses(PROMPTS, max_new_tokens=12) == reference["plain"]
    tiny_setup.intervention_applier.apply_direction_intervention(
        DirectionVector(t.randn(64, generator=t.Generator().manual_seed(1)), 1, -1, 0), "ablate", layers=[0, 1])
    try:
        assert ev.generate_responses(PROMPTS, max_new_tokens=12) == reference["ablate"]
    finally:
        tiny_setup.intervention_applier.clear_interventions()


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_log_odds_metric_is_batch_invariant(tiny_setup, batch_size):
    from evaluation import BigEvaluator
    make = lambda bs: BigEvaluator(tiny_setup, gen_batch_size=bs, target_tokens=["I", " I"])
    assert make(batch_size)._log_odds_metric(PROMPTS) == pytest.approx(make(1)._log_odds_metric(PROMPTS), abs=1e-5)


def test_oom_mid_run_recovers_identical_output(tiny_setup, reference, monkeypatch):
    """A batch that runs out of memory is split and retried: same text, and the
    split is counted (so the report can flag it)."""
    import evaluation
    real = evaluation.generate_with_hooks
    def flaky(model, tok, fmt, prompts, max_new_tokens=64):
        if len(prompts) > 4:
            raise RuntimeError("MPS backend out of memory")
        return real(model, tok, fmt, prompts, max_new_tokens=max_new_tokens)
    monkeypatch.setattr(evaluation, "generate_with_hooks", flaky)
    ev = _evaluator(tiny_setup, 16)
    assert ev.generate_responses(PROMPTS, max_new_tokens=12) == reference["plain"]
    assert ev.oom_splits == 3  # 16 -> 8+8 -> each 8 -> 4+4


def test_auto_resolves_to_power_of_two_on_real_model(tiny_setup):
    ev = _evaluator(tiny_setup, "auto")
    bs = ev.resolve_batch_size(PROMPTS, 512)
    assert bs >= 1 and bs & (bs - 1) == 0 and bs <= batching.MAX_AUTO_BATCH_SIZE
