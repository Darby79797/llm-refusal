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
    _joint_ablation_basis,
    analyze_direction_subspace,
    measure_interference,
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

    def test_hooks_cleared_after_exception_in_measure_interference(self, monkeypatch):
        """If evaluation raises mid-way through measure_interference, the intervention
        hooks registered for the in-flight ablation must still be removed (finally)."""
        from interventions import ModelInterventionApplier
        import evaluation

        model = self._make_stub_model()
        applier = ModelInterventionApplier(model)

        direction = _make_direction(t.randn(8))
        concept = MagicMock()
        concept.name = "test_concept"
        concept.detection_phrases = ["x"]
        concept.detection_fn = None
        concept.judge_prompt = None
        concept.eval_data_fn = MagicMock(return_value=(["p1", "p2"], []))

        tokenizer = MagicMock()
        prompt_formatter = MagicMock()

        call_count = {"n": 0}

        def flaky_evaluate_detection_rate(self, prompts, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise RuntimeError("simulated evaluation failure")
            return 0.5

        monkeypatch.setattr(
            evaluation.BigEvaluator, "evaluate_detection_rate", flaky_evaluate_detection_rate
        )

        with pytest.raises(RuntimeError, match="simulated evaluation failure"):
            measure_interference(
                model, tokenizer, applier, prompt_formatter,
                [direction], [concept], max_prompts=2,
            )

        # The finally block must have run: no hooks left registered on the applier.
        assert applier.intervention_hooks == []
