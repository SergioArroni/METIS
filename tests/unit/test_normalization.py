"""Tests for metis.shared.normalization — scoring normalization functions.

WHY: Wrong normalization = wrong scores. This is the single most critical
mathematical module. Every normalizer must:
  - Return values in [0, 1]
  - Map "best" raw values to 1.0
  - Handle edge cases (NaN, Inf, negative)
  - Be monotonic in the expected direction

Uses hypothesis for property-based testing of invariants.
"""

import math

import pytest
from hypothesis import given, settings, strategies as st

from metis.shared.normalization import (
    METRIC_NORMALIZATION_MAP,
    clamp,
    get_normalization_params,
    get_normalizer,
    normalize_bounded_distance,
    normalize_dcr,
    normalize_delta,
    normalize_metric_value,
    normalize_ml_loss,
    normalize_mutual_information,
    normalize_similarity,
    normalize_unbounded_distance,
    normalize_values_batch,
)

# =============================================================================
# clamp() — foundational utility
# =============================================================================


class TestClamp:
    def test_in_range_unchanged(self):
        assert clamp(0.5) == 0.5

    def test_zero(self):
        assert clamp(0.0) == 0.0

    def test_one(self):
        assert clamp(1.0) == 1.0

    def test_negative_clipped(self):
        assert clamp(-0.1) == 0.0

    def test_above_one_clipped(self):
        assert clamp(1.5) == 1.0

    def test_nan_returns_zero(self):
        assert clamp(float("nan")) == 0.0

    def test_inf_returns_zero(self):
        assert clamp(float("inf")) == 0.0

    def test_neg_inf_returns_zero(self):
        assert clamp(float("-inf")) == 0.0


# =============================================================================
# Individual normalization functions
# =============================================================================


class TestNormalizeBoundedDistance:
    """For metrics like KS, Hellinger in [0,1] where 0 = perfect match."""

    def test_zero_is_best(self):
        assert normalize_bounded_distance(0.0) == 1.0

    def test_one_is_worst(self):
        assert normalize_bounded_distance(1.0) == 0.0

    def test_midpoint(self):
        assert normalize_bounded_distance(0.5) == pytest.approx(0.5)

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_output_in_range(self, x):
        result = normalize_bounded_distance(x)
        assert 0.0 <= result <= 1.0

    @given(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0))
    def test_monotonically_decreasing(self, a, b):
        """Lower raw distance → higher normalized score."""
        if a < b:
            assert normalize_bounded_distance(a) >= normalize_bounded_distance(b)


class TestNormalizeUnboundedDistance:
    """For metrics like Wasserstein, MMD where 0 = best but unbounded above."""

    def test_zero_is_best(self):
        assert normalize_unbounded_distance(0.0) == 1.0

    def test_large_value_near_zero(self):
        assert normalize_unbounded_distance(100.0) < 0.01

    def test_scale_parameter(self):
        # With larger scale, decay is slower
        slow = normalize_unbounded_distance(1.0, {"scale": 10.0})
        fast = normalize_unbounded_distance(1.0, {"scale": 0.1})
        assert slow > fast

    def test_negative_scale_uses_default(self):
        """Defensive: negative/zero scale falls back to 1.0."""
        result = normalize_unbounded_distance(1.0, {"scale": -1.0})
        assert 0.0 <= result <= 1.0

    @given(st.floats(min_value=0.0, max_value=1000.0))
    def test_output_in_range(self, x):
        assert 0.0 <= normalize_unbounded_distance(x) <= 1.0


class TestNormalizeSimilarity:
    """For metrics already in [0,1] where 1 = best (Pearson, AUC, etc.)."""

    def test_identity_for_valid_range(self):
        assert normalize_similarity(0.7) == pytest.approx(0.7)

    def test_clips_above_one(self):
        assert normalize_similarity(1.5) == 1.0

    def test_clips_below_zero(self):
        assert normalize_similarity(-0.1) == 0.0


class TestNormalizeDelta:
    """For delta metrics where 0 = perfect match, uses 1/(1+|x|)."""

    def test_zero_is_best(self):
        assert normalize_delta(0.0) == 1.0

    def test_positive_delta(self):
        assert normalize_delta(1.0) == pytest.approx(0.5)

    def test_negative_delta_same_as_positive(self):
        """Uses absolute value."""
        assert normalize_delta(-2.0) == normalize_delta(2.0)

    @given(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    def test_output_in_range(self, x):
        assert 0.0 <= normalize_delta(x) <= 1.0


class TestNormalizeMutualInformation:
    """MI is unbounded ≥ 0, uses value/(value+scale)."""

    def test_zero_mi(self):
        assert normalize_mutual_information(0.0) == 0.0

    def test_high_mi(self):
        # With default scale=1.0: 10/(10+1) ≈ 0.909
        result = normalize_mutual_information(10.0)
        assert result == pytest.approx(10.0 / 11.0)

    def test_scale_parameter(self):
        # Higher scale → lower result for same MI value
        low_scale = normalize_mutual_information(1.0, {"mi_scale": 0.5})
        high_scale = normalize_mutual_information(1.0, {"mi_scale": 5.0})
        assert low_scale > high_scale


class TestNormalizeDCR:
    """DCR: higher = more private, uses 1-exp(-value/scale)."""

    def test_zero_dcr(self):
        assert normalize_dcr(0.0) == pytest.approx(0.0)

    def test_large_dcr(self):
        result = normalize_dcr(100.0)
        assert result > 0.99

    @given(st.floats(min_value=0.0, max_value=1000.0))
    def test_output_in_range(self, x):
        assert 0.0 <= normalize_dcr(x) <= 1.0


class TestNormalizeMLLoss:
    """ML loss: lower = better, uses 1/(1+value)."""

    def test_zero_loss_is_best(self):
        assert normalize_ml_loss(0.0) == 1.0

    def test_high_loss(self):
        assert normalize_ml_loss(9.0) == pytest.approx(0.1)


# =============================================================================
# Registry dispatch
# =============================================================================


class TestNormalizeMetricValue:
    """End-to-end dispatch from metric_id to correct normalizer."""

    def test_known_metric(self):
        # fidelity.ks uses BOUNDED_DISTANCE → 1 - value
        result = normalize_metric_value("fidelity.ks", 0.1)
        assert result == pytest.approx(0.9)

    def test_unbounded_with_params(self):
        # fidelity.wasserstein uses UNBOUNDED_DISTANCE with scale=0.5
        result = normalize_metric_value("fidelity.wasserstein", 0.0)
        assert result == 1.0

    def test_unknown_metric_defaults_to_similarity(self):
        result = normalize_metric_value("unknown.metric", 0.8)
        assert result == pytest.approx(0.8)

    def test_param_override(self):
        # Override the default scale for wasserstein
        default_result = normalize_metric_value("fidelity.wasserstein", 1.0)
        custom_result = normalize_metric_value("fidelity.wasserstein", 1.0, {"scale": 10.0})
        assert custom_result > default_result  # slower decay with larger scale


class TestGetNormalizer:
    def test_all_mapped_metrics_resolve(self):
        """Every entry in METRIC_NORMALIZATION_MAP must resolve to a callable."""
        for metric_id in METRIC_NORMALIZATION_MAP:
            fn = get_normalizer(metric_id)
            assert callable(fn)

    def test_short_name_resolution(self):
        """'ks' should resolve to the same normalizer as 'fidelity.ks'."""
        fn_full = get_normalizer("fidelity.ks")
        fn_short = get_normalizer("ks")
        # Both should produce same result
        assert fn_full(0.3, None) == fn_short(0.3, None)


class TestNormalizeValuesBatch:
    def test_batch_normalization(self):
        raw = {"col_a": 0.1, "col_b": 0.5, "col_c": 0.9}
        result = normalize_values_batch(raw, "fidelity.ks")
        assert result["col_a"] == pytest.approx(0.9)
        assert result["col_c"] == pytest.approx(0.1)

    def test_nan_preserved(self):
        raw = {"col_a": 0.5, "col_b": float("nan")}
        result = normalize_values_batch(raw, "fidelity.ks")
        assert result["col_a"] == pytest.approx(0.5)
        assert math.isnan(result["col_b"])


# =============================================================================
# Property-based: all normalizers return [0,1] for reasonable inputs
# =============================================================================


class TestNormalizationProperties:
    """Verify invariants across ALL normalization types with hypothesis."""

    @given(st.floats(min_value=0.0, max_value=100.0))
    @settings(max_examples=200)
    def test_all_normalizers_return_valid_range(self, value):
        """Every normalizer must return a value in [0, 1] for non-negative input."""
        for metric_id in METRIC_NORMALIZATION_MAP:
            params = get_normalization_params(metric_id)
            result = normalize_metric_value(metric_id, value, params)
            assert 0.0 <= result <= 1.0, f"{metric_id} returned {result} for input {value}"
