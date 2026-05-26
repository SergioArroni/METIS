"""Tests for metis.shared.aggregation_registry — aggregation functions.

WHY: Aggregation functions combine metric scores into family/composite scores.
Wrong aggregation = wrong final evaluation. NaN-safety is critical because
metrics can fail and produce NaN — the aggregator must NOT propagate failures
to entire family scores.

Uses hypothesis for invariant testing.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from metis.shared.aggregation_registry import (
    AGGREGATION_FUNCTIONS,
    get_aggregation_category,
    get_aggregation_function,
    list_aggregation_functions,
)

# =============================================================================
# Registry API
# =============================================================================


class TestRegistryAPI:
    def test_get_known_function(self):
        fn = get_aggregation_function("mean")
        assert callable(fn)

    def test_get_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown aggregation function"):
            get_aggregation_function("nonexistent")

    def test_error_message_lists_available(self):
        with pytest.raises(KeyError, match="Available:"):
            get_aggregation_function("bad_name")

    def test_list_functions_sorted(self):
        funcs = list_aggregation_functions()
        assert funcs == sorted(funcs)
        assert "mean" in funcs
        assert "ssd" in funcs

    def test_ssd_with_custom_risk_aversion(self):
        """Parameterized SSD must respect risk_aversion."""
        fn_default = get_aggregation_function("ssd")
        fn_high_risk = get_aggregation_function("ssd", risk_aversion=10.0)
        values = [0.3, 0.7, 0.9]
        # Higher risk aversion penalizes low values more
        assert fn_high_risk(values) <= fn_default(values)

    def test_get_aggregation_category(self):
        assert get_aggregation_category("mean") == "basic"
        assert get_aggregation_category("ssd") == "stochastic_dominance"
        assert get_aggregation_category("unknown_fn") == "other"


# =============================================================================
# Individual aggregation functions — correctness
# =============================================================================


class TestBasicAggregators:
    def test_mean(self):
        fn = AGGREGATION_FUNCTIONS["mean"]
        assert fn([0.2, 0.4, 0.6]) == pytest.approx(0.4)

    def test_median(self):
        fn = AGGREGATION_FUNCTIONS["median"]
        assert fn([0.1, 0.5, 0.9]) == pytest.approx(0.5)

    def test_min(self):
        fn = AGGREGATION_FUNCTIONS["min"]
        assert fn([0.3, 0.1, 0.9]) == pytest.approx(0.1)

    def test_max(self):
        fn = AGGREGATION_FUNCTIONS["max"]
        assert fn([0.3, 0.1, 0.9]) == pytest.approx(0.9)


class TestPercentileAggregators:
    def test_percentile_25(self):
        fn = AGGREGATION_FUNCTIONS["percentile_25"]
        result = fn([0.0, 0.25, 0.5, 0.75, 1.0])
        assert result == pytest.approx(0.25)

    def test_percentile_75(self):
        fn = AGGREGATION_FUNCTIONS["percentile_75"]
        result = fn([0.0, 0.25, 0.5, 0.75, 1.0])
        assert result == pytest.approx(0.75)


class TestRobustAggregators:
    def test_trimmed_mean_10(self):
        fn = AGGREGATION_FUNCTIONS["trimmed_mean_10"]
        # Should be more robust to outliers than mean
        values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0]
        result = fn(values)
        mean = AGGREGATION_FUNCTIONS["mean"](values)
        # Trimmed mean should be closer to 0.5 than regular mean
        assert abs(result - 0.5) <= abs(mean - 0.5) + 1e-10


class TestMeanVariants:
    def test_geometric_mean_positive(self):
        fn = AGGREGATION_FUNCTIONS["geometric_mean"]
        result = fn([0.5, 0.5, 0.5])
        assert result == pytest.approx(0.5)

    def test_geometric_mean_with_zero(self):
        """Geometric mean with zero → returns 0.0 (graceful handling)."""
        fn = AGGREGATION_FUNCTIONS["geometric_mean"]
        assert fn([0.5, 0.0, 0.8]) == 0.0

    def test_harmonic_mean_positive(self):
        fn = AGGREGATION_FUNCTIONS["harmonic_mean"]
        result = fn([1.0, 1.0, 1.0])
        assert result == pytest.approx(1.0)

    def test_harmonic_mean_with_zero(self):
        fn = AGGREGATION_FUNCTIONS["harmonic_mean"]
        assert fn([0.5, 0.0, 0.8]) == 0.0


class TestStochasticDominanceAggregators:
    def test_fsd_is_mean(self):
        """FSD score is equivalent to arithmetic mean."""
        fn = AGGREGATION_FUNCTIONS["fsd"]
        values = [0.3, 0.5, 0.7]
        assert fn(values) == pytest.approx(np.mean(values), abs=1e-6)

    def test_ssd_penalizes_variance(self):
        """SSD with risk aversion should penalize unequal distributions."""
        fn = AGGREGATION_FUNCTIONS["ssd"]
        uniform = [0.5, 0.5, 0.5]
        varied = [0.1, 0.5, 0.9]
        # Same mean but SSD should prefer uniform (less risky)
        assert fn(uniform) >= fn(varied)


# =============================================================================
# NaN safety — critical for production robustness
# =============================================================================


class TestNaNSafety:
    """Every aggregator must handle NaN/Inf gracefully.

    Risk: A single failed metric (NaN) should NOT corrupt the entire
    family score. Aggregators must strip NaN before computing.
    """

    @pytest.mark.parametrize("agg_name", list(AGGREGATION_FUNCTIONS.keys()))
    def test_nan_values_stripped(self, agg_name):
        """NaN values should be ignored, not propagate."""
        fn = AGGREGATION_FUNCTIONS[agg_name]
        values = [0.5, float("nan"), 0.7]
        result = fn(values)
        # Result should be computed from [0.5, 0.7] only
        assert not math.isnan(result)

    @pytest.mark.parametrize("agg_name", list(AGGREGATION_FUNCTIONS.keys()))
    def test_inf_values_stripped(self, agg_name):
        fn = AGGREGATION_FUNCTIONS[agg_name]
        values = [0.5, float("inf"), 0.7]
        result = fn(values)
        assert math.isfinite(result)

    @pytest.mark.parametrize("agg_name", list(AGGREGATION_FUNCTIONS.keys()))
    def test_all_nan_returns_nan(self, agg_name):
        """When ALL values are NaN, result should be NaN (nothing to aggregate)."""
        fn = AGGREGATION_FUNCTIONS[agg_name]
        result = fn([float("nan"), float("nan")])
        assert math.isnan(result)

    @pytest.mark.parametrize("agg_name", list(AGGREGATION_FUNCTIONS.keys()))
    def test_empty_list_returns_nan(self, agg_name):
        fn = AGGREGATION_FUNCTIONS[agg_name]
        result = fn([])
        assert math.isnan(result)


# =============================================================================
# Property-based: aggregators on valid [0,1] inputs
# =============================================================================


class TestAggregationProperties:
    """Invariants that must hold for all aggregators on metric-valid inputs."""

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=20))
    @settings(max_examples=200)
    @pytest.mark.parametrize(
        "agg_name",
        [
            "mean",
            "median",
            "min",
            "max",
            "trimmed_mean_10",
            "percentile_25",
            "percentile_75",
            "fsd",
            "ssd",
        ],
    )
    def test_output_in_valid_range(self, agg_name, values):
        """All aggregators should return values in [0,1] for inputs in [0,1]."""
        fn = AGGREGATION_FUNCTIONS[agg_name]
        result = fn(values)
        assert 0.0 <= result <= 1.0 + 1e-9, f"{agg_name}({values}) = {result}"

    def test_single_element(self):
        """Aggregating a single value should return that value (for most aggregators)."""
        for name in ["mean", "median", "min", "max", "fsd"]:
            fn = AGGREGATION_FUNCTIONS[name]
            assert fn([0.42]) == pytest.approx(0.42, abs=1e-6)
