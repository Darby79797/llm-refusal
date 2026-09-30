"""Tests for cross_concept.py bug fixes:

1. Hook-lifecycle safety: clear_interventions() must run even if evaluation raises.
2. Joint (order-independent) multi-direction ablation, replacing sequential
   per-direction projection removal.
3. Uncentered-SVD subspace analysis (no NaN for near-identical vectors, measures the
   subspace spanned from the origin rather than variance around the centroid).
"""
import numpy as np
import pytest
import torch as t
from unittest.mock import MagicMock

from datatypes import DirectionVector
from cross_concept import (
    CellRunner,
    _joint_ablation_basis,
    analyze_direction_subspace,
    measure_interference,
    run_multi_ablation,
)


def _make_direction(vec, layer=0):
    return DirectionVector(vector=vec, layer=layer, position_index=-1, score=0.0)


def _remove_span(x: t.Tensor, basis: t.Tensor) -> t.Tensor:
    """Same math as apply_joint_ablation's hooks: x - (x @ Q) @ Q.T."""
    return x - t.matmul(t.matmul(x, basis), basis.T)


# ============================================================
# Joint ablation: order independence
# ============================================================

class TestJointAblationOrderIndependence:
    def test_order_independent_and_zeroes_projection(self):
        t.manual_seed(0)
        dim = 16
        a = t.randn(dim, dtype=t.float64)
        a_unit = a / a.norm()

        # Build b with cosine similarity ~0.6 to a.
        noise = t.randn(dim, dtype=t.float64)
        noise_orth = noise - (noise @ a_unit) * a_unit
        noise_orth = noise_orth / noise_orth.norm()
        cos = 0.6
        b = cos * a_unit + (1 - cos ** 2) ** 0.5 * noise_orth

        dA = _make_direction(a)
        dB = _make_direction(b)
        assert (dA.unit @ dB.unit).item() == pytest.approx(0.6, abs=1e-6)

        x = t.randn(5, dim, dtype=t.float64)

        basis_ab = _joint_ablation_basis([dA, dB])
        basis_ba = _joint_ablation_basis([dB, dA])

        out_ab = _remove_span(x, basis_ab)
        out_ba = _remove_span(x, basis_ba)

        assert t.allclose(out_ab, out_ba, atol=1e-8)

        # Residual has ~zero projection onto both original directions.
        assert t.allclose(out_ab @ dA.unit.to(t.float64), t.zeros(5, dtype=t.float64), atol=1e-8)
        assert t.allclose(out_ab @ dB.unit.to(t.float64), t.zeros(5, dtype=t.float64), atol=1e-8)


# ============================================================
# Joint ablation: rank deficiency
# ============================================================

class TestJointAblationRankDeficiency:
    def test_duplicate_direction_equals_single(self):
        t.manual_seed(1)
        dim = 12
        a = t.randn(dim, dtype=t.float64)
        dA = _make_direction(a)

        x = t.randn(4, dim, dtype=t.float64)

        basis_single = _joint_ablation_basis([dA])
        basis_dup = _joint_ablation_basis([dA, dA])

        assert basis_single.shape[1] == 1
        assert basis_dup.shape[1] == 1  # rank-deficient pair collapses, not double-counted

        out_single = _remove_span(x, basis_single)
        out_dup = _remove_span(x, basis_dup)
        assert t.allclose(out_single, out_dup, atol=1e-8)

    def test_near_parallel_directions_collapse_to_rank_1(self):
        t.manual_seed(2)
        dim = 10
        a = t.randn(dim, dtype=t.float64)
        # b is a scaled copy of a (same ray up to sign/scale is not required here,
        # but exact scalar multiples span a 1D subspace).
        b = a * 3.0
        dA = _make_direction(a)
        dB = _make_direction(b)
        basis = _joint_ablation_basis([dA, dB])
        assert basis.shape[1] == 1


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

    def test_near_identical_vectors_no_nan(self):
        """Regression check for the sklearn-PCA 0/0 NaN bug on near-identical inputs."""
        v1 = t.zeros(8); v1[0] = 1.0
        v2 = t.zeros(8); v2[0] = 1.0 + 1e-9
        d1 = _make_direction(v1)
        d2 = _make_direction(v2, layer=1)
        explained = analyze_direction_subspace([d1, d2])
        assert not np.isnan(explained).any()


# ============================================================
# Hook-lifecycle safety
# ============================================================

class TestHookSafety:
    def _make_stub_model(self, n_layers=3):
        model = MagicMock()
        model.dtype = t.float32
        model.model.layers = [MagicMock() for _ in range(n_layers)]
        for layer in model.model.layers:
            layer.self_attn = MagicMock()
            layer.mlp = MagicMock()
            layer.register_forward_pre_hook = MagicMock(return_value=MagicMock())
            layer.self_attn.register_forward_hook = MagicMock(return_value=MagicMock())
            layer.mlp.register_forward_hook = MagicMock(return_value=MagicMock())
        return model

    def _make_runner(self, monkeypatch, n_concepts=1, fail_after=None):
        """CellRunner over a stub model, with _measure stubbed (no generation). Returns
        (runner, applier, calls) where calls counts _measure invocations."""
        from interventions import ModelInterventionApplier

        model = self._make_stub_model()
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

    def test_joint_ablation_hooks_cleared_after_exception(self, monkeypatch):
        runner, _, _ = self._make_runner(monkeypatch, n_concepts=3, fail_after=0)
        layer = runner.intervention_applier.transformer_layers[0]
        with pytest.raises(RuntimeError, match="simulated evaluation failure"):
            runner.run(["c0", "c1"], "c2")
        handle = layer.register_forward_pre_hook.return_value
        assert handle.remove.called

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
