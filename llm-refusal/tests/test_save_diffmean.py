"""save_diffmean.py: aligned halves of a concept's pairs, and the norm-matched mean of unit vectors."""
import importlib.util
import os

import torch as t


def _mod():
    spec = importlib.util.spec_from_file_location(
        "save_diffmean", os.path.join(os.path.dirname(__file__), "..", "scripts", "save_diffmean.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_halves_are_aligned_and_disjoint():
    sd = _mod()
    pos, neg = [f"p{i}" for i in range(6)], [f"n{i}" for i in range(6)]
    a, b = sd.half((pos, neg), "a"), sd.half((pos, neg), "b")
    assert a == (pos[:3], neg[:3]) and b == (pos[3:], neg[3:]) and sd.half((pos, neg), "all") == (pos, neg)


def test_mean_of_units_weights_members_equally_and_takes_the_norm():
    sd = _mod()
    v = sd.mean_of_units([t.tensor([10.0, 0.0]), t.tensor([0.0, 0.1])], norm=4.0)
    assert abs(float(v.norm()) - 4.0) < 1e-6 and abs(float(v[0] - v[1])) < 1e-6
