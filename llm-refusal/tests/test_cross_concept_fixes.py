"""cross_concept.py and the multi-direction ablation it relies on:

1. Hook-lifecycle safety: clear_interventions() must run even if evaluation raises.
2. Joint (order-independent) multi-direction ablation via interventions.py's
   [k, d] ablate path, not sequential per-direction projection removal.
3. Uncentered-SVD subspace analysis (no NaN for near-identical vectors, measures the
   subspace spanned from the origin rather than variance around the centroid).
"""
import numpy as np
import pytest
import torch as t
from unittest.mock import MagicMock

from datatypes import DirectionVector
from interventions import ModelInterventionApplier
from cross_concept import (
    CellRunner,
    analyze_direction_subspace,
    measure_interference,
    run_multi_ablation,
)
from tests.fakes import ResidualModel, no_hooks


def _make_direction(vec, layer=0):
    return DirectionVector(vector=vec, layer=layer, position_index=-1, score=0.0)


def _ablated_output(model, direction, x):
    applier = ModelInterventionApplier(model)
    with applier.intervened(direction, "ablate"):
        return model(x)


def _cos_pair(dim, cos, seed):
    t.manual_seed(seed)
    a = t.randn(dim, dtype=t.float64)
    a_unit = a / a.norm()
    noise = t.randn(dim, dtype=t.float64)
    noise_orth = noise - (noise @ a_unit) * a_unit
    noise_orth = noise_orth / noise_orth.norm()
    return a, cos * a_unit + (1 - cos ** 2) ** 0.5 * noise_orth


# ============================================================
# Joint ablation: order independence
# ============================================================

class TestJointAblationOrderIndependence:
    def test_order_independent_and_zeroes_projection(self):
        dim = 16
        a, b = _cos_pair(dim, 0.6, seed=0)
        dA, dB = _make_direction(a), _make_direction(b)
        assert (dA.unit @ dB.unit).item() == pytest.approx(0.6, abs=1e-6)

        model = ResidualModel(dim, dtype=t.float64)
        x = t.randn(5, dim, dtype=t.float64)
        out_ab = _ablated_output(model, t.stack([a, b]), x)
        out_ba = _ablated_output(model, t.stack([b, a]), x)
        assert t.allclose(out_ab, out_ba, atol=1e-6)

        # Residual has ~zero projection onto both original directions.
        assert t.allclose(out_ab @ dA.unit, t.zeros(5, dtype=t.float64), atol=1e-6)  # basis is fp32
        assert t.allclose(out_ab @ dB.unit, t.zeros(5, dtype=t.float64), atol=1e-6)  # basis is fp32

        # Sequential single-direction ablation is order-dependent; the joint one is not that.
        applier = ModelInterventionApplier(model)
        applier.apply_direction_intervention(dA, "ablate")
        applier.apply_direction_intervention(dB, "ablate")
        try:
            out_seq = model(x)
        finally:
            applier.clear_interventions()
        assert not t.allclose(out_seq, out_ab, atol=1e-3)


# ============================================================
# [k, d] ablate: k=1 equals the vector path; invalid stacks are refused
# ============================================================

class TestStackedAblate:
    @pytest.mark.parametrize("dtype", [t.float64, t.float32, t.bfloat16])
    def test_k1_stack_matches_vector_exactly(self, dtype):
        dim = 12
        model = ResidualModel(dim, dtype=dtype, seed=1)
        v = t.randn(dim)
        x = t.randn(4, dim).to(dtype)
        out_vec = _ablated_output(model, _make_direction(v), x)
        out_k1 = _ablated_output(model, v.unsqueeze(0), x)
        out_raw = _ablated_output(model, v, x)
        assert t.equal(out_vec, out_k1)
        assert t.equal(out_vec, out_raw)

    def test_linearly_dependent_stack_raises(self):
        a = t.randn(10, dtype=t.float64)
        applier = ModelInterventionApplier(ResidualModel(10, dtype=t.float64))
        with pytest.raises(ValueError, match="linearly dependent"):
            applier.apply_direction_intervention(t.stack([a, 3.0 * a]), "ablate")
        assert applier.intervention_hooks == []

    def test_stack_only_supports_ablate(self):
        applier = ModelInterventionApplier(ResidualModel(10, dtype=t.float64))
        with pytest.raises(ValueError, match="only supports 'ablate'"):
            applier.apply_direction_intervention(t.randn(2, 10, dtype=t.float64), "add")
        assert applier.intervention_hooks == []


# ============================================================
# Subspace analysis (uncentered SVD)
# ============================================================

class TestSubspaceAnalysis:
    def test_identical_vectors_no_nan_rank_1(self):
        vec = t.randn(20)
        d1 = _make_direction(vec.clone())
        d2 = _make_direction(vec.clone(), layer=1)
        explained = analyze_direction_subspace([d1, d2])
        assert not np.isnan(explained).any()
        assert explained[0] == pytest.approx(1.0, abs=1e-6)
        rank = int(np.sum(explained > 1e-6))
        assert rank == 1

    def test_orthogonal_vectors_split_50_50_rank_2(self):
        v1 = t.zeros(10); v1[0] = 1.0
        v2 = t.zeros(10); v2[1] = 1.0
        d1 = _make_direction(v1)
        d2 = _make_direction(v2, layer=1)
        explained = analyze_direction_subspace([d1, d2])
        assert not np.isnan(explained).any()
        assert explained[0] == pytest.approx(0.5, abs=1e-6)
        assert explained[1] == pytest.approx(0.5, abs=1e-6)
        rank = int(np.sum(explained > 1e-6))
        assert rank == 2


# ============================================================
# Hook-lifecycle safety
# ============================================================

class TestHookSafety:
    def _make_runner(self, monkeypatch, n_concepts=1, fail_after=None):
        """CellRunner over a stub model, with _measure stubbed (no generation). Returns
        (runner, applier, calls) where calls counts _measure invocations."""
        from interventions import ModelInterventionApplier

        model = ResidualModel(8, n_layers=3)
        applier = ModelInterventionApplier(model)
        concepts, directions = [], []
        for i in range(n_concepts):
            concept = MagicMock()
            concept.name = f"c{i}"
            concept.detection_phrases = ["x"]
            concept.detection_fn = None
            concept.judge_prompt = None
            concept.target_tokens = None
            concept.eval_data_fn = MagicMock(return_value=(["p1", "p2"], []))
            concepts.append(concept)
            directions.append(_make_direction(t.randn(8)))

        calls = {"n": 0}

        def fake_measure(self, measured, key):
            calls["n"] += 1
            if fail_after is not None and calls["n"] > fail_after:
                raise RuntimeError("simulated evaluation failure")
            return {"rate": 0.5, "log_odds": None, "degenerate_rate": 0.0, "n": 2}

        monkeypatch.setattr(CellRunner, "_measure", fake_measure)
        runner = CellRunner(model, MagicMock(), applier, MagicMock(), directions, concepts,
                            gen_batch_size=2)
        return runner, applier, calls

    def test_hooks_cleared_after_exception_in_measure_interference(self, monkeypatch):
        """If evaluation raises mid-way through measure_interference, the intervention
        hooks registered for the in-flight ablation must still be removed (finally)."""
        runner, applier, _ = self._make_runner(monkeypatch, fail_after=1)
        with pytest.raises(RuntimeError, match="simulated evaluation failure"):
            measure_interference(runner, ["c0"])
        assert applier.intervention_hooks == []
        assert no_hooks(applier.transformer_layers)

    def test_joint_ablation_hooks_cleared_after_exception(self, monkeypatch):
        runner, _, _ = self._make_runner(monkeypatch, n_concepts=3, fail_after=0)
        with pytest.raises(RuntimeError, match="simulated evaluation failure"):
            runner.run(["c0", "c1"], "c2")
        assert no_hooks(runner.intervention_applier.transformer_layers)

    def test_run_multi_ablation_reuses_interference_cells(self, monkeypatch):
        """3 concepts: 3 baselines + 9 single ablations + 3 joint ablations = 15 cells.
        The multi-ablation test must not regenerate baselines or single ablations."""
        runner, _, calls = self._make_runner(monkeypatch, n_concepts=3)
        names = ["c0", "c1", "c2"]
        measure_interference(runner, names)
        assert calls["n"] == 12
        for name in names:
            run_multi_ablation(runner, [n for n in names if n != name], name)
        assert calls["n"] == 15
        # Joint cells are keyed order-independently.
        assert runner.key(["c1", "c0"], "c2") == runner.key(["c0", "c1"], "c2")
