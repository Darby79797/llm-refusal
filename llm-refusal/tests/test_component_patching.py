"""component_patching.py: necessity / sufficiency shares and the sublayer output swap helpers."""
import importlib.util
import math
import os

import torch as t


def _mod():
    spec = importlib.util.spec_from_file_location(
        "component_patching", os.path.join(os.path.dirname(__file__), "..", "scripts", "component_patching.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_effect_share_necessity_and_sufficiency():
    cp = _mod()
    # clean +8, pushed -2: the push moved log-odds by -10
    assert cp.effect_share("necessity", 8.0, 8.0, -2.0) == 1.0      # patching clean back undoes all of it
    assert cp.effect_share("necessity", -2.0, 8.0, -2.0) == 0.0
    assert cp.effect_share("sufficiency", 3.0, 8.0, -2.0) == 0.5    # the pushed output alone brings half
    assert math.isnan(cp.effect_share("necessity", 1.0, 1.0, 1.2))


def test_output_helpers_handle_tuple_and_tensor_outputs():
    cp = _mod()
    x, w = t.ones(2), t.zeros(1)
    assert cp._out((x, w)) is x and cp._out(x) is x
    new = t.full((2,), 3.0)
    r = cp._replace((x, w), new)
    assert r[0] is new and r[1] is w
    assert cp._replace(x, new) is new
