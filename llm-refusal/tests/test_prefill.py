"""prefill:<text> condition: sets and clears the evaluator's prefill; generate_responses passes it as every row's
prefix and returns prefix + continuation."""
import importlib.util
import os

import evaluation


def _js():
    spec = importlib.util.spec_from_file_location(
        "jbb_safety", os.path.join(os.path.dirname(__file__), "..", "scripts", "jbb_safety.py"))
    js = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(js)
    return js


def test_prefill_condition_sets_and_clears_the_evaluators_prefill():
    js = _js()

    class Run:
        class ev:
            prefill = None
    with js.condition_context(Run, "prefill:I'm not aware of any information about"):
        assert Run.ev.prefill == "I'm not aware of any information about"
    assert Run.ev.prefill is None


def test_generate_responses_prefixes_each_row_and_returns_prefix_plus_continuation(monkeypatch):
    seen = []

    def fake(model, tok, fmt, chunk, max_new_tokens=64, prefixes=None):
        seen.append(prefixes)
        return [f"<{p}>" for p in chunk]
    monkeypatch.setattr(evaluation, "generate_with_hooks", fake)

    class Fw:
        prompt_formatter = None
    ev = evaluation.BigEvaluator.__new__(evaluation.BigEvaluator)
    ev.model, ev.tokenizer, ev.framework = None, None, Fw()
    ev.prefill = "I cannot"
    out = ev.generate_responses(["a", "b"], batch_size=2)
    assert seen == [["I cannot", "I cannot"]] and out == ["I cannot<a>", "I cannot<b>"]
    ev.prefill = None
    assert ev.generate_responses(["a"], batch_size=1) == ["<a>"] and seen[-1] is None  # not passed at all
