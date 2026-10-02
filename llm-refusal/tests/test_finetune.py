"""Completion batches, completion CE, and the low-rank adapters (capability.py, finetune.py).

A tiny random Qwen2 model with the real Qwen2.5 tokenizer (cached, offline): no
checkpoint weights are loaded.
"""
import pytest
import torch as t
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from capability import completion_ce
from finetune import adapted, adapter_sites, train_adapters
from formatting import ChatPromptFormatter


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen2.5 tokenizer not cached")


@pytest.fixture
def model(tokenizer):
    t.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return Qwen2ForCausalLM(cfg).float().eval()


PROMPTS = ["Give three tips for staying healthy.", "Name a colour."]
COMPLETIONS = ["Eat well, sleep, and exercise.", "Blue."]


def test_completion_labels_cover_only_the_completion(tokenizer):
    fmt = ChatPromptFormatter(tokenizer)
    enc = fmt.format_with_completions(PROMPTS, COMPLETIONS)
    for j, (p, c) in enumerate(zip(PROMPTS, COMPLETIONS)):
        prompt_len = int(fmt.format_batch([p])['attention_mask'].sum())
        c_ids = tokenizer.encode(c, add_special_tokens=False)
        labels = enc['labels'][j]
        assert (labels[:prompt_len] == -100).all()
        assert labels[prompt_len:prompt_len + len(c_ids)].tolist() == c_ids
        assert (labels[prompt_len + len(c_ids):] == -100).all()
        assert int(enc['attention_mask'][j].sum()) == prompt_len + len(c_ids)
    # Right-padded: every row's real tokens start at 0.
    assert (enc['attention_mask'][:, 0] == 1).all()


def test_raw_text_rows_score_every_token_but_the_first(tokenizer):
    enc = ChatPromptFormatter(tokenizer).format_with_completions(None, ["hello there world"])
    n = int(enc['attention_mask'].sum())
    assert enc['labels'][0, 0] == -100 and (enc['labels'][0, 1:n] != -100).all()


def test_completion_ce_matches_full_logits(model, tokenizer):
    fmt = ChatPromptFormatter(tokenizer)
    got = completion_ce(model, fmt, PROMPTS, COMPLETIONS, batch_size=2)
    enc = fmt.format_with_completions(PROMPTS, COMPLETIONS)
    with t.no_grad():
        logits = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                       position_ids=enc['position_ids']).logits
    want = t.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                         enc['labels'][:, 1:].reshape(-1), ignore_index=-100)
    assert got == pytest.approx(want.item(), rel=1e-5)
    # Batch size doesn't change a token-weighted mean.
    assert completion_ce(model, fmt, PROMPTS, COMPLETIONS, batch_size=1) == pytest.approx(got, rel=1e-5)


def test_adapter_starts_as_identity_trains_and_restores(model, tokenizer):
    fmt = ChatPromptFormatter(tokenizer)
    ids = fmt.format_batch(PROMPTS)
    with t.no_grad():
        before = model(**ids).logits
    down = model.model.layers[0].mlp.down_proj
    sites = adapter_sites(model, [0], ["down_proj"])
    examples = list(zip(PROMPTS, COMPLETIONS))
    with adapted(model, sites, rank=1, seed=0) as adapters:
        with t.no_grad():
            assert t.allclose(model(**ids).logits, before)   # U = 0 at init
        hist = train_adapters(model, fmt, adapters, examples, steps=30, lr=1e-2, batch_size=2, seed=0)
        assert hist[-1]["loss"] < hist[0]["loss"]
        assert adapters[0].U.norm() > 0
        assert all(not p.requires_grad for n, p in model.named_parameters() if not n.endswith((".U", ".V")))
    assert model.model.layers[0].mlp.down_proj is down
    with t.no_grad():
        assert t.allclose(model(**ids).logits, before)


def test_adapter_sites_reject_unknown_modules(model):
    with pytest.raises(ValueError):
        adapter_sites(model, [0], ["not_a_proj"])


def test_completion_ce_chunking_matches_unchunked(model, tokenizer, monkeypatch):
    """Applying the unembedding a few tokens at a time gives the same CE."""
    import capability
    fmt = ChatPromptFormatter(tokenizer)
    whole = completion_ce(model, fmt, PROMPTS, COMPLETIONS, batch_size=2)
    monkeypatch.setattr(capability, "CE_CHUNK_TOKENS", 3)
    assert completion_ce(model, fmt, PROMPTS, COMPLETIONS, batch_size=2) == pytest.approx(whole, rel=1e-6)


def test_reserve_bytes_shrinks_the_auto_batch_budget(monkeypatch):
    import batching
    monkeypatch.setattr(batching, "device_total_bytes", lambda device: 40 * 2**30)
    monkeypatch.setattr(batching, "weight_bytes", lambda model: 16 * 2**30)
    monkeypatch.setattr(batching, "estimate_forward_bytes_per_row", lambda *a: 2**30)
    fake = type("M", (), {"config": None, "dtype": t.bfloat16, "device": t.device("cpu")})()
    # budget 0.5 * (40 - 16) = 12 GiB -> 8 rows of 1 GiB; reserving 12 GiB: 0.5 * 12 = 6 GiB -> 4 rows
    assert batching.resolve_forward_batch_size("auto", fake, 512) == 8
    assert batching.resolve_forward_batch_size("auto", fake, 512, reserve_bytes=12 * 2**30) == 4


def test_edit_bytes_counts_every_edited_tensor(model):
    from orthogonalize import edit_bytes, residual_writers
    want = sum(m.weight.numel() * m.weight.element_size() for m, _ in residual_writers(model))
    assert edit_bytes(model) == want


def test_micro_batching_gives_the_same_update(model, tokenizer):
    """Gradient accumulation over micro-batches = one full-batch step (token-weighted)."""
    fmt = ChatPromptFormatter(tokenizer)
    examples = list(zip(PROMPTS, COMPLETIONS)) * 2
    sites = adapter_sites(model, [1], ["down_proj"])
    trained = []
    for micro in (None, 1, 3):
        with adapted(model, sites, rank=2, seed=0) as adapters:
            hist = train_adapters(model, fmt, adapters, examples, steps=3, lr=1e-2, batch_size=4, seed=0,
                                  micro_batch_size=micro)
            trained.append((adapters[0].U.detach().clone(), hist[-1]["loss"]))
    for U, loss in trained[1:]:
        assert t.allclose(U, trained[0][0], atol=1e-5)
        assert loss == pytest.approx(trained[0][1], rel=1e-5)


def test_micro_batches_split_long_rows():
    from finetune import _micro_batches
    spans = lambda ls, r, t: [(s.start, s.stop) for s in _micro_batches(ls, r, t)]
    assert spans([60, 70, 80, 90], 2, 256) == [(0, 2), (2, 4)]          # typical rows: pairs
    assert spans([60, 314, 80, 90], 2, 256) == [(0, 1), (1, 2), (2, 4)]  # the 314-token row alone
    assert spans([400], 2, 256) == [(0, 1)]                              # one row always allowed
    assert spans([10] * 5, 8, 10_000) == [(0, 5)]


def test_token_capped_micro_batches_give_the_same_update(model, tokenizer, monkeypatch):
    import finetune
    fmt = ChatPromptFormatter(tokenizer)
    examples = list(zip(PROMPTS, COMPLETIONS)) * 2
    sites = adapter_sites(model, [1], ["down_proj"])
    out = []
    for per_row in (10_000, 1):   # no token cap vs every row on its own
        monkeypatch.setattr(finetune, "MICRO_TOKENS_PER_ROW", per_row)
        with adapted(model, sites, rank=2, seed=0) as adapters:
            train_adapters(model, fmt, adapters, examples, steps=3, lr=1e-2, batch_size=4, seed=0, micro_batch_size=4)
            out.append(adapters[0].U.detach().clone())
    assert t.allclose(out[0], out[1], atol=1e-5)


def _reference_update(model, fmt, sites, examples, micro):
    with adapted(model, sites, rank=2, seed=0) as adapters:
        hist = train_adapters(model, fmt, adapters, examples, steps=1, lr=1e-2, batch_size=4, seed=0,
                              micro_batch_size=micro)
        return adapters[0].U.detach().clone(), hist


def _flaky_forward(monkeypatch, model, should_fail):
    """Make the decoder raise an MPS-style OOM when should_fail(call_no, input_ids) is true."""
    base = model.get_decoder()
    real, calls = base.forward, []
    def forward(*a, **k):
        calls.append(k["input_ids"].shape[0])
        if should_fail(len(calls), k["input_ids"]):
            raise RuntimeError("MPS backend out of memory (simulated)")
        return real(*a, **k)
    monkeypatch.setattr(base, "forward", forward)
    return calls


def test_oom_retries_only_the_failed_micro_batch(model, tokenizer, monkeypatch):
    """Finished micro-batches are kept; the failed one is split; the update is unchanged."""
    import finetune
    monkeypatch.setattr(finetune, "MICRO_TOKENS_PER_ROW", 10_000)
    fmt = ChatPromptFormatter(tokenizer)
    examples = list(zip(PROMPTS, COMPLETIONS)) * 2
    sites = adapter_sites(model, [1], ["down_proj"])
    want, _ = _reference_update(model, fmt, sites, examples, micro=2)
    calls = _flaky_forward(monkeypatch, model, lambda n, ids: n == 2)   # 2nd micro-batch OOMs once
    got, hist = _reference_update(model, fmt, sites, examples, micro=2)
    assert calls == [2, 2, 1, 1]          # rows 0-1 once; rows 2-3 failed, then split into singles
    assert hist[0]["oom_splits"] == 1 and "skipped_rows" not in hist[0]
    assert t.allclose(got, want, atol=1e-5)


def test_row_too_big_on_its_own_is_skipped(model, tokenizer, monkeypatch):
    """A single row that always OOMs is dropped from the step, not a failure; the loss is
    the token-mean CE over the other rows."""
    import finetune
    monkeypatch.setattr(finetune, "MICRO_TOKENS_PER_ROW", 10_000)
    fmt = ChatPromptFormatter(tokenizer)
    examples = list(zip(PROMPTS, COMPLETIONS)) + [("Name a fruit.", "Apple."), ("Name a tree.", "Oak.")]
    poison = tokenizer.encode("Name a fruit.", add_special_tokens=False)
    def contains_poison(ids):
        row = ids[0].tolist()
        return any(row[i:i + len(poison)] == poison for i in range(len(row)))
    calls = _flaky_forward(monkeypatch, model, lambda n, ids: ids.shape[0] == 1 and contains_poison(ids)
                           or ids.shape[0] > 1 and any(contains_poison(ids[j:j + 1]) for j in range(ids.shape[0])))
    _, hist = _reference_update(model, fmt, sites := adapter_sites(model, [1], ["down_proj"]), examples, micro=4)
    assert hist[0]["skipped_rows"] == 1
    others = [e for e in examples if e[0] != "Name a fruit."]
    monkeypatch.undo()
    # U starts at 0, so step 1's loss is the unadapted model's CE on the rows used.
    want = completion_ce(model, fmt, [p for p, _ in others], [c for _, c in others], batch_size=4)
    assert hist[0]["loss"] == pytest.approx(want, rel=1e-4)
