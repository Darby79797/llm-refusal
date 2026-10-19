"""claim_trajectory.py: answer-token buckets and reading a condition's spec as a prompt-only push."""
import importlib.util
import math
import os

import torch as t


def _mod():
    spec = importlib.util.spec_from_file_location(
        "claim_trajectory", os.path.join(os.path.dirname(__file__), "..", "scripts", "claim_trajectory.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bucket_means_are_over_answer_tokens_by_1_based_position():
    ct = _mod()
    proj = t.arange(30, dtype=t.float)              # value = absolute position
    b = ct.bucket_means(proj, start=10, end=30)     # answer tokens at positions 10..29 (k = 1..20)
    assert b[0] == (10 + 11 + 12 + 13) / 4          # k 1-4
    assert b[1] == sum(range(14, 26)) / 12          # k 5-16
    assert b[2] == sum(range(26, 30)) / 4           # k 17-20 (bucket 17-64 truncated at the end)
    assert math.isnan(b[3])                         # no k >= 65


def test_prompt_only_push_reads_kind_sign_and_strength():
    ct = _mod()
    ss = lambda s: (s.rpartition("*")[0], float(s.rpartition("*")[2])) if "*" in s else (s, 1.0)
    assert ct.prompt_only_push("baseline", ss) is None
    assert ct.prompt_only_push("add:results/u-direction", ss) == ("results/u-direction", 1.0)
    assert ct.prompt_only_push("addpos:all:results/u-direction*2", ss) == ("results/u-direction", 2.0)
    assert ct.prompt_only_push("subpos:all+gen16:results/u-direction", ss) == ("results/u-direction", -1.0)
    assert ct.prompt_only_push("sub:results/u-direction*0.5", ss) == ("results/u-direction", -0.5)
