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
