"""push_capability.py: prompts, GSM8K answer extraction, multiple-choice scoring (no model)."""
import importlib.util
import json
import os

import torch as t


def _mod():
    spec = importlib.util.spec_from_file_location(
        "push_capability", os.path.join(os.path.dirname(__file__), "..", "scripts", "push_capability.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gsm8k_extract_prefers_the_answer_line():
    pc = _mod()
    assert pc.gsm8k_extract("3 + 4 = 7 apples, then 2 more.\nAnswer: 1,250") == "1250"
    assert pc.gsm8k_extract("So she pays $18.\n**Answer:** $18.") == "18"
    assert pc.gsm8k_extract("It costs 12 dollars, so 24 in total.") == "24"
    assert pc.gsm8k_extract("I can't help with that.") is None
    assert pc.gsm8k_correct("18", "18") and pc.gsm8k_correct("18.0", "18") and not pc.gsm8k_correct("17", "18")
    assert not pc.gsm8k_correct(None, "18") and not pc.gsm8k_correct("-", "18")


def test_mc_prompt_letters_every_option():
    pc = _mod()
    p = pc.mc_prompt({"question": "2+2?", "choices": ["3", "4", "5"], "answer": 1})
    assert "A. 3\nB. 4\nC. 5" in p and "letter" in p


class _Tok:
    def encode(self, s, add_special_tokens=False):
        return [{"A": 10, " A": 20, "B": 11, " B": 21, "C": 12, " C": 22, "D": 13, " D": 23, "E": 14, " E": 24}[s]]


def test_letter_ids_and_scores_take_each_letters_best_variant():
    pc = _mod()
    ids = pc.letter_ids(_Tok())
    assert ids[0] == [10, 20] and ids[3] == [13, 23]
    logits = t.zeros(30)
    logits[21] = 5.0   # " B" is the top letter
    logits[10] = 1.0
    logits[13] = 9.0   # D, but the item has only three options
    s = pc.mc_scores(logits, ids, 3)
    assert len(s) == 3 and max(range(3), key=s.__getitem__) == 1


def test_benchmark_items_file_shape():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_items.json")
    if not os.path.exists(path):
        return  # fetched by scripts/fetch_benchmark_items.py; data/ is not in git
    d = json.load(open(path))
    assert set(d) == {"mmlu", "arc", "gsm8k"}
    for x in d["mmlu"] + d["arc"]:
        assert 0 <= x["answer"] < len(x["choices"]) <= 5
    assert all(float(x["answer"]) == float(x["answer"]) for x in d["gsm8k"])


def test_jbb_conditions_dedupes_specs_and_renames_clashes(tmp_path, monkeypatch):
    pc = _mod()
    (tmp_path / "results" / "analysis").mkdir(parents=True)
    write = lambda tag, conds: json.dump({"conditions": {k: {"spec": v} for k, v in conds.items()}},
                                         open(tmp_path / f"results/analysis/M-jbb-safety-{tag}.json", "w"))
    write("one", {"baseline": "baseline", "lim": "sub:L"})
    write("two", {"baseline": "baseline", "lim": "sub:L*0.5", "rhat": "sub:R", "same": "sub:L"})
    monkeypatch.chdir(tmp_path)
    assert pc.jbb_conditions("M", ["one", "two", "missing"]) == [
        ("baseline", "baseline"), ("lim", "sub:L"), ("lim-two", "sub:L*0.5"), ("rhat", "sub:R")]


def test_gsm8k_scores_accuracy_and_missing_answer_lines():
    pc = _mod()

    class _Run:
        class ev:
            @staticmethod
            def generate_responses(prompts, max_new_tokens):
                return ["2 + 2 = 4\nAnswer: 4", "It is 5, I think.", "I can't help with that."]

        class fw:
            class tokenizer:
                @staticmethod
                def encode(s, add_special_tokens=False):
                    return [ord(c) for c in s]

    r = pc.gsm8k(_Run, [{"question": "q", "answer": "4"}, {"question": "q", "answer": "5"},
                        {"question": "q", "answer": "1"}], 8)
    assert r["correct"] == [1, 1, 0] and abs(r["no_answer"] - 2 / 3) < 1e-9
