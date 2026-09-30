"""tools/: run index parsing (file names, logs), rescoring, and a report build over a
synthetic results root. No model is loaded."""
import json
import os

import pytest

from tools import look, report
from tools.runs import EVAL_RE, SEARCH_RE, detector, load_index, wilson

GENS = {
    "baseline": [{"prompt": f"p{i}", "response": "I can’t help with that." if i < 9 else "Sure, here.",
                  "detected": i < 8, "degenerate": False, "nll": 0.5} for i in range(10)],
    "global_ablation": [{"prompt": f"p{i}", "response": "Sure, here.", "detected": False,
                         "degenerate": i == 0, "nll": 1.0} for i in range(10)],
}
LOG = """2026-09-24 - INFO - Filtering: reusing cached result x.json (positive 90→71, negative 64→63)
==================== COMPREHENSIVE EVALUATION REPORT ====================
  gen_batch_size: 64, max_new_tokens: 64, dtype: torch.bfloat16
--- BASELINE ---
  refusal_rate_on_positive_prompts: 0.8000
  log_odds_metric: 4.0621
--- GLOBAL ABLATION ---
  refusal_rate_on_positive_prompts: 0.0000
  log_odds_metric: -8.1545
======================================================================
"""
SEARCH_LOG = "--- Strictly Selected Direction (Met All Criteria) ---\n2026 - INFO - Layer: 21, Position: -4\n"


@pytest.fixture
def root(tmp_path):
    (tmp_path / "Qwen2.5-3B-Instruct-refusal-evaluate-L21-P-4-generations.json").write_text(json.dumps(GENS))
    (tmp_path / "Qwen2.5-3B-Instruct-refusal-evaluate-L21-P-4.log").write_text(LOG)
    (tmp_path / "filtered-Qwen2.5-3B-Instruct-sycophancy-evaluate-L17-P-6-generations.json").write_text(json.dumps(GENS))
    (tmp_path / "Qwen2.5-3B-Instruct-refusal-search-scores.csv").write_text(
        "layer,position,bypass_score,induce_score,induce_global_score,kl_score\n21,-4,-8.0,3.0,1.0,0.01\n21,-1,,2.0,1.0,0.02\n")
    (tmp_path / "Qwen2.5-3B-Instruct-refusal-search.log").write_text(SEARCH_LOG)
    (tmp_path / "Qwen2.5-3B-Instruct-refusal-direction.json").write_text(json.dumps({"layer": 21, "position_index": -4, "score": -8.0}))
    return str(tmp_path)


def test_filename_patterns():
    m = EVAL_RE.match("Meta-Llama-3-8B-Instruct-refusal_arditi_exact-evaluate-L12-P-5-T512-generations.json")
    assert (m["model"], m["concept"], m["tag"], m["variant"]) == ("Meta-Llama-3-8B-Instruct", "refusal_arditi_exact", "L12-P-5-T512", None)
    m = EVAL_RE.match("filtered-Qwen2.5-0.5B-Instruct-sycophancy-evaluate-L16-P-1-generations.json")
    assert (m["variant"], m["model"], m["concept"]) == ("filtered", "Qwen2.5-0.5B-Instruct", "sycophancy")
    m = EVAL_RE.match("Llama-2-7b-chat-hf-refusal-evaluate-L12-P-1-generations.json")
    assert (m["variant"], m["model"]) == (None, "Llama-2-7b-chat-hf")
    assert SEARCH_RE.match("Qwen2.5-7B-Instruct-hedging-search-scores.csv")["concept"] == "hedging"


def test_index_parses_runs_logs_and_search(root):
    ix = load_index([root])
    run = ix.find_evals("3B", "refusal")[0]
    assert (run.layer, run.pos, run.variant) == (21, -4, "")
    assert run.header == {"gen_batch_size": "64", "max_new_tokens": "64", "dtype": "bfloat16"}
    assert run.filter == {"pos_in": 90, "pos_out": 71, "neg_in": 64, "neg_out": 63}
    base, abl = run.conditions["baseline"], run.conditions["global_ablation"]
    assert (base.k, base.n, base.log_odds) == (8, 10, 4.0621)
    assert abl.log_odds == -8.1545 and abl.degenerate == pytest.approx(0.1)
    assert run.effect("global_ablation") == "down"
    assert ix.find_evals("3B", "sycophancy", variant="filtered")[0].variant == "filtered"
    s = ix.find_search("3B", "refusal")
    assert (s.selected, s.tier) == ((21, -4), "strict")
    assert len(s.rows) == 2
    assert ix.directions[("Qwen2.5-3B-Instruct", "refusal")]["layer"] == 21


def test_exact_tag_beats_substring(root, tmp_path):
    (tmp_path / "Qwen2.5-3B-Instruct-refusal-evaluate-L21-P-4-T128-generations.json").write_text(json.dumps(GENS))
    ix = load_index([root])
    assert [r.tag for r in ix.find_evals("3B", "refusal", "L21-P-4")] == ["L21-P-4"]


def test_detector_normalizes_apostrophes():
    detect = detector("refusal")
    assert detect("I can’t help with that.")
    assert not detect("Sure, here.")


def test_llamaguard_raw_verdicts_count_as_unsafe():
    from tools.runs import is_unsafe
    assert is_unsafe("unsafe\nS2") and is_unsafe(True) and not is_unsafe("safe") and not is_unsafe(None)


def test_wilson_bounds():
    lo, hi = wilson(0, 99)
    assert lo == 0 and hi == pytest.approx(0.0373, abs=1e-3)


def test_look_commands_run(root, capsys):
    for argv in (["digest"], ["run", "3B", "refusal"], ["gens", "3B", "refusal", "--flip", "baseline:global_ablation", "--rescore"],
                 ["search", "3B", "refusal", "--metric", "induce"]):
        look.main(["--root", root] + argv)
    out = capsys.readouterr().out
    assert "Qwen2.5-3B" in out and "selected (21, -4) via strict" in out


def test_report_builds_and_rescores(root, tmp_path):
    out = str(tmp_path / "report.html")
    report.main(["--root", root, "--out", out])
    html = open(out).read()
    assert "/*__DATA__*/null" not in html
    data = json.loads(html.split("const DATA = ")[1].split(";\nconst $")[0])
    run = next(r for r in data["runs"] if r["id"] == "Qwen2.5-3B-Instruct/refusal/L21-P-4")
    assert run["relabeled"] == 1            # prompt 8: curly-apostrophe refusal saved as undetected
    assert run["gens"]["baseline"][8]["was"] is False
