"""CAA (caa.py): tokenization matches the reference implementation, padding and
batching don't change results, and steering touches only positions from the boundary."""
import pytest
import torch as t

import caa
from formatting import ChatPromptFormatter
from tests.fakes import cached_tokenizer, tiny_qwen2

QUESTION = "Is the sky blue?\n\nChoices:\n (A) Yes\n (B) No"


@pytest.fixture(scope="module")
def tiny():
    """Random 3-layer Qwen2 with the real Qwen2.5 tokenizer (fp32, CPU)."""
    tok = cached_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    model = tiny_qwen2(tok, num_hidden_layers=3)
    fmt = ChatPromptFormatter(tok)
    return model, tok, fmt


def _runner(tiny, bs):
    model, tok, fmt = tiny
    return caa.CAA(model, tok, fmt, model.model.layers, batch_size=bs)


def _items(n=5):
    qs = [QUESTION, "Short?\n (A) x\n (B) y",
          "A much longer question about many things, with plenty of words to pad the batch.\n (A) one\n (B) two",
          "Q4\n (A) a\n (B) b", "Q5 is here\n (A) c\n (B) d"][:n]
    return [{"question": q, "answer_matching_behavior": "(A)" if i % 2 else "(B)",
             "answer_not_matching_behavior": "(B)" if i % 2 else "(A)"} for i, q in enumerate(qs)]


class TestEncoder:
    def test_llama2_matches_reference_tokenization(self):
        """Reference: tokenizer.encode(f"[INST] {q.strip()} [/INST] {answer}") with BOS,
        vector read at index -2, steering from the last token of "[/INST]"."""
        tok = cached_tokenizer("meta-llama/Llama-2-7b-chat-hf")
        enc = caa.CAAEncoder(tok, ChatPromptFormatter(tok)).encode(QUESTION, "A")
        reference = tok.encode(f"[INST] {QUESTION.strip()} [/INST] (A)")
        assert enc.ids == reference
        assert tok.convert_ids_to_tokens(enc.ids[-2]) == "A"
        assert tok.convert_ids_to_tokens(enc.ids[enc.boundary]) == "]"
        assert tok.convert_ids_to_tokens(enc.ids[enc.boundary + 1]) == "▁("

    @pytest.mark.parametrize("name", ["Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"])
    def test_letter_is_own_token_after_template(self, name):
        tok = cached_tokenizer(name)
        encoder = caa.CAAEncoder(tok, ChatPromptFormatter(tok))
        for letter in "AB":
            enc = encoder.encode(QUESTION, letter)
            assert tok.decode([enc.ids[-2]]).strip() == letter
            # boundary = last token of the templated prompt, just before " ("
            assert tok.decode(enc.ids[enc.boundary + 1:]).strip() == f"({letter})"
        assert encoder.letter_ids["A"] != encoder.letter_ids["B"]


class TestBatching:
    def test_vectors_batch_invariant(self, tiny):
        items = _items()
        v1 = _runner(tiny, 1).compute_vectors(items)
        v4 = _runner(tiny, 4).compute_vectors(items)
        assert v1.shape == (3, 32)
        assert t.allclose(v1, v4, atol=1e-5)

    def test_ab_probs_batch_invariant(self, tiny):
        items = _items()
        vec = t.randn(32)
        for layer, m in [(None, 0.0), (1, 2.0)]:
            r1 = _runner(tiny, 1).ab_probs(items, layer, vec if layer is not None else None, m)
            r4 = _runner(tiny, 4).ab_probs(items, layer, vec if layer is not None else None, m)
            for a, b in zip(r1, r4):
                assert a["p_a"] == pytest.approx(b["p_a"], abs=1e-5)
                assert a["p_match"] == pytest.approx(b["p_match"], abs=1e-5)

    def test_zero_multiplier_is_baseline_and_steering_moves(self, tiny):
        items = _items()
        r = _runner(tiny, 4)
        base = r.ab_probs(items)
        zero = r.ab_probs(items, 1, t.randn(32) * 50, 0.0)
        steered = r.ab_probs(items, 1, t.randn(32) * 50, 1.0)
        assert [x["p_a"] for x in base] == [x["p_a"] for x in zero]
        # (random model: p(A) itself is ~1/vocab, so compare the normalized ratio)
        assert any(abs(x["p_match"] - y["p_match"]) > 1e-3 for x, y in zip(base, steered))

    def test_cached_vectors_match_reference(self, tiny):
        """KV-cached prefix + one-token letter suffix == CAA's full forward read at -2."""
        items = _items()
        for bs in (1, 4):
            r = _runner(tiny, bs)
            assert t.allclose(r.compute_vectors(items), r.compute_vectors_full(items), atol=1e-5)

    def test_cached_ab_matches_reference_with_steering(self, tiny):
        """Cached sweep == one uncached forward per item steering positions >= boundary,
        for several conditions sharing one prefix cache (the cache is cropped between)."""
        items = _items()
        vec = t.randn(32)
        conds = [(None, None, 0.0), (1, vec, 3.0), (0, vec, -2.0), (1, vec, 3.0)]
        for bs in (1, 3):
            sweep = _runner(tiny, bs).ab_sweep(items, conds)
            for (layer, v, m), got in zip(conds, sweep):
                ref = _runner(tiny, 1).ab_probs_full(items, layer, v, m)
                for a, b in zip(got, ref):
                    assert a["p_match"] == pytest.approx(b["p_match"], abs=1e-5)
                    assert a["p_a"] == pytest.approx(b["p_a"], rel=1e-4)
        assert [x["p_a"] for x in sweep[1]] == [x["p_a"] for x in sweep[3]]   # crop restored the prefix

    def test_steering_touches_only_suffix(self, tiny):
        """Steering from the boundary on: the prefix activations the cache holds are the
        clean ones, so a steered condition leaves the next unsteered one unchanged."""
        items = _items(2)
        r = _runner(tiny, 2)
        a, _, b = r.ab_sweep(items, [(None, None, 0.0), (1, t.randn(32) * 50, 1.0), (None, None, 0.0)])
        assert [x["p_a"] for x in a] == [x["p_a"] for x in b]
        assert not tiny[0].model.layers[1]._forward_hooks

    def test_token_budget_batches(self, tiny):
        r = _runner(tiny, 64)
        r.max_batch_tokens = 100
        batches = r._batches([10, 60, 20, 30, 40])
        assert sorted(sum(batches, [])) == [0, 1, 2, 3, 4]
        lengths = [10, 60, 20, 30, 40]
        for b in batches:
            assert max(lengths[i] for i in b) * len(b) <= 100 or len(b) == 1
        assert batches[0][0] == 0            # length-sorted


def test_normalize_across_behaviors():
    vecs = {"a": t.randn(4, 8, dtype=t.float64) * 3, "b": t.randn(4, 8, dtype=t.float64)}
    out = caa.normalize_across_behaviors(vecs)
    mean = (vecs["a"].norm(dim=-1) + vecs["b"].norm(dim=-1)) / 2
    for v in out.values():
        assert t.allclose(v.norm(dim=-1), mean)
    # direction unchanged
    assert t.allclose(out["a"] / out["a"].norm(dim=-1, keepdim=True),
                      vecs["a"] / vecs["a"].norm(dim=-1, keepdim=True))


def test_cosine_table_uses_matched_layer():
    v = t.zeros(4, 8, dtype=t.float64)
    v[:, 0] = 1.0
    v[2] = 0.0
    v[2, 1] = 1.0                        # CAA layer 2 points along e1
    ours = t.zeros(8); ours[1] = 5.0     # our layer 3 = CAA layer 2
    row = caa.cosine_table({"x": v}, {"refusal": (3, ours)})["refusal"]["x"]
    assert row["matched_caa_layer"] == 2
    assert row["cos_matched"] == pytest.approx(1.0)
    assert row["best_caa_layer"] == 2


def test_datasets_present_and_well_formed():
    for b in caa.BEHAVIORS:
        test = caa.load_ab(b, "test")
        gen = caa.load_ab(b, "generate")
        assert len(test) == 50 and len(gen) >= 290
        test_q = {x["question"] for x in test}
        assert not test_q & {x["question"] for x in gen}, f"{b}: test/generate overlap"
        for x in test + gen:
            assert caa._letter(x["answer_matching_behavior"]) != caa._letter(x["answer_not_matching_behavior"])


def test_multi_option_item_scores_both_metrics(tiny):
    """survival-instinct items labelled e.g. (E)/(A): p_match uses the two given letters;
    p_match_caa reproduces the reference plotting code, which scores them 0."""
    item = {"question": "Q\n (A) a\n (B) b\n (C) c\n (D) d\n (E) e",
            "answer_matching_behavior": "(E)", "answer_not_matching_behavior": "(A)"}
    r = _runner(tiny, 1).ab_probs([item])[0]
    assert r["p_match_caa"] == 0.0
    assert r["p_match"] == pytest.approx(r["p_matching_letter"] / (r["p_matching_letter"] + r["p_a"]))
