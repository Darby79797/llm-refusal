"""dirspecs.py (the direction-spec grammar shared by the analysis scripts) and deflection.py (the refuse / moralise /
other / comply labels). Synthetic directions in a temp dir; no model is loaded."""
import json

import pytest
import torch as t

import deflection
import dirspecs
from datatypes import DirectionVector

D = 16


@pytest.fixture
def stems(tmp_path):
    """r (the refusal direction) and five saved directions with a shared component and some r̂ in them."""
    g = t.Generator().manual_seed(0)
    r = DirectionVector(vector=t.randn(D, generator=g) * 3, layer=4, position_index=-1, score=0)
    common = t.randn(D, generator=g)
    out = []
    for i in range(5):
        v = common + 0.3 * t.randn(D, generator=g) + 0.5 * r.vector
        stem = str(tmp_path / f"d{i}-direction")
        DirectionVector(vector=v, layer=6 + i, position_index=-1, score=0).save(stem)
        out.append(stem)
    return r, out


def u(r):
    return dirspecs.unit(r.vector.float())


def test_rhat_stem_and_random(stems):
    r, s = stems
    assert dirspecs.load_spec("rhat", r) is r
    d = dirspecs.load_spec(s[0], r)
    assert d.layer == 6 and t.equal(d.vector, DirectionVector.load(s[0]).vector)
    a = dirspecs.load_spec(f"random@{s[0]}", r, lambda spec: 1)
    b = dirspecs.load_spec(f"random@{s[0]}", r, lambda spec: 2)
    assert a.layer == 6 and t.isclose(a.vector.norm(), d.vector.norm())   # norm-matched, at the stem's coordinates
    assert not t.allclose(a.vector, b.vector)                             # the seed comes from seed_of


def test_perp_removes_rhat_from_stems_and_stacks(stems):
    r, s = stems
    p = dirspecs.load_spec(f"perp:{s[0]}", r)
    assert isinstance(p, DirectionVector) and abs(float(p.vector @ u(r))) < 1e-5
    stack = dirspecs.load_spec(f"perp:span:{s[0]}+{s[1]}", r)
    assert stack.shape == (2, D) and (stack @ u(r)).abs().max() < 1e-5


def test_span_amp_and_randspan_shapes(stems):
    r, s = stems
    assert dirspecs.load_spec(f"span:{s[0]}+{s[1]}+{s[2]}", r).shape == (3, D)
    both = dirspecs.load_spec(f"rhat&{s[1]}", r)
    assert both.shape == (2, D) and t.allclose(both[0], r.vector.float())
    assert dirspecs.load_spec(f"randspan:4@{s[0]}", r).shape == (4, D)


def test_pcs_and_pc_select_the_same_orthonormal_axes(stems):
    r, s = stems
    pool = "+".join(f"perp:{x}" for x in s)
    top = dirspecs.load_spec(f"pcs:3@{pool}", r)
    assert t.allclose(top @ top.T, t.eye(3), atol=1e-5)
    picked = dirspecs.load_spec(f"pc:1/3@{pool}", r)
    assert t.allclose(picked, top[[0, 2]])
    assert (top @ u(r)).abs().max() < 1e-5                    # a pool of perp: directions stays ⊥ r̂


def test_tilt_has_the_exact_cosine_and_no_rhat(stems):
    r, s = stems
    pool = "+".join(f"perp:{x}" for x in s)
    a0 = dirspecs.load_spec(f"pc:1@{pool}", r)[0]
    for c in (0.1, 0.4):
        v = dirspecs.load_spec(f"tilt:{c}:1@pc:1@{pool}", r, lambda spec: 7)[0]
        assert abs(float(v.norm()) - 1) < 1e-5
        assert abs(abs(float(v @ a0)) - c) < 1e-5 and abs(float(v @ u(r))) < 1e-5


def test_signed_axes_point_to_the_pool_side(stems):
    r, s = stems
    pool = "+".join(s)
    (k, a1), = dirspecs.signed_pc_axes(pool, r, [1])
    V = t.stack([dirspecs.unit(DirectionVector.load(x).vector.float()) for x in s])
    assert k == 1 and float((V @ a1).sum()) > 0
    neg = [x + "-neg" for x in s]
    for x, y in zip(s, neg):
        d = DirectionVector.load(x)
        DirectionVector(vector=-d.vector, layer=d.layer, position_index=-1, score=0).save(y)
    (_, b1), = dirspecs.signed_pc_axes("+".join(neg), r, [1])
    assert t.allclose(b1, -a1, atol=1e-5)                    # the sign follows the pool, not the SVD's arbitrary one


def test_save_pc_axes_writes_loadable_unit_stems(stems, tmp_path):
    r, s = stems
    pool = "+".join(f"perp:{x}" for x in s)
    out = dirspecs.save_pc_axes(pool, r, [1, 2], layer=9, out_prefix=str(tmp_path / "pcax-m"))
    assert out == [str(tmp_path / "pcax-m-PC1-direction"), str(tmp_path / "pcax-m-PC2-direction")]
    d1 = DirectionVector.load(out[0])
    assert (d1.layer, d1.position_index) == (9, -1) and abs(float(d1.vector.norm()) - 1) < 1e-5
    assert t.allclose(d1.vector.abs(), dirspecs.load_spec(f"pc:1@{pool}", r)[0].abs(), atol=1e-6)
    assert json.load(open(tmp_path / "pcax-m-axes.json"))["stems"] == out


def test_push_vector_moves_the_projection_by_scale_times_gap():
    axis = t.tensor([3.0, 4.0, 0.0])
    v = dirspecs.push_vector(axis, gap=2.5, scale=2.0)
    assert t.isclose(v @ dirspecs.unit(axis), t.tensor(5.0))
    assert t.isclose(dirspecs.push_vector(axis, -1.0, 1.0) @ dirspecs.unit(axis), t.tensor(-1.0))  # sign kept


def test_as_edit_vector_casts_stems_and_stacks(stems):
    r, s = stems
    assert dirspecs.as_edit_vector(r, t.bfloat16).dtype == t.bfloat16
    assert dirspecs.as_edit_vector(t.zeros(2, D), t.float16).shape == (2, D)


def test_deflection_labels():
    assert deflection.label(None, True, "x") is None
    assert deflection.label("unsafe\nS2", True, "I can't") == "comply"       # LG2 unsafe wins over a phrase
    assert deflection.label("safe", True, "I can't help") == "refuse"
    assert deflection.label("safe", False, "Bribery is illegal and unethical.") == "moralise"
    assert deflection.label("safe", False, "Gifts to officials are common.") == "other"


def test_deflection_conditions_cover_pushes_and_jbb():
    import prompts
    harmful, harmless = prompts.create_refusal_eval_data()
    d = {"n_harmful": 2, "n_harmless": 3, "results": {"clean": {"none": {
        "texts": ["a", "b"], "labels": [True, False],
        "pushes": {"pc:1@x@5:2": {"texts": ["c", "d", "e"], "labels": [False] * 3}}}}}}
    got = list(deflection.conditions(d))
    assert [g[0] for g in got] == ["clean|none", "clean|none|push:pc:1@x@5:2"]
    assert got[0][1] == harmful[:2] and got[1][1] == harmless[:3]        # push responses are judged against harmless
    assert got[1][2] == ["c", "d", "e"] and got[1][4] is None
    jbb = {"n": 2, "conditions": {"base": {"responses": ["x", "y"], "detected": [True, False],
                                           "llamaguard": ["safe", None]},
                                  "half": {"responses": ["x", None], "detected": [True, False], "llamaguard": [None, None]}}}
    got = list(deflection.conditions(jbb))
    assert [g[0] for g in got] == ["jbb|base"]                            # incomplete generations are skipped
    assert got[0][1] == prompts.create_arditi_replication_eval_data()[0][:2]
    assert got[0][4] == ["safe", None]                                     # saved verdicts are reused
