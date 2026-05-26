"""Tests for fidelity metrics computation (parametrized)."""

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.registry import get_metric_registry


@pytest.fixture(scope="module")
def registry():
    """Get the global metric registry with all metrics registered."""
    return get_metric_registry()


@pytest.fixture
def real_numeric():
    """Real numeric series for testing."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(50, 10, size=200), name="test_col")


@pytest.fixture
def synth_numeric():
    """Synthetic numeric series (slight shift)."""
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(52, 11, size=200), name="test_col")


@pytest.fixture
def identical_numeric():
    """Identical real and synthetic series (perfect score expected)."""
    rng = np.random.default_rng(42)
    data = rng.normal(50, 10, size=200)
    return pd.Series(data, name="test_col"), pd.Series(data.copy(), name="test_col")


@pytest.fixture
def real_categorical():
    """Real categorical series."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.choice(["A", "B", "C", "D"], size=200), name="test_col")


@pytest.fixture
def synth_categorical():
    """Synthetic categorical series (similar distribution)."""
    rng = np.random.default_rng(99)
    return pd.Series(rng.choice(["A", "B", "C", "D"], size=200), name="test_col")


# Marginal tails metrics that work on numeric columns
MARGINAL_TAIL_METRICS = [
    "fidelity.ks",
    "fidelity.wasserstein",
    "fidelity.anderson_darling",
    "fidelity.hellinger",
    "fidelity.kde_ise",
    "fidelity.delta_exceedance",
]

# Marginal scale metrics
MARGINAL_SCALE_METRICS = [
    "fidelity.delta_mean",
    "fidelity.delta_median",
    "fidelity.delta_iqr",
    "fidelity.delta_mad",
    "fidelity.cohens_d",
]

# Coverage metrics (work on categorical or numeric)
COVERAGE_METRICS = [
    "fidelity.tvd",
    "fidelity.js",
    "fidelity.kl",
    "fidelity.psi",
    "fidelity.entropy_delta",
    "fidelity.gini_delta",
]


class TestMarginalTailMetrics:
    """Test that marginal tail metrics produce valid results."""

    @pytest.mark.parametrize("metric_id", MARGINAL_TAIL_METRICS)
    def test_metric_returns_float(self, registry, metric_id, real_numeric, synth_numeric):
        metric_cls = registry.get(metric_id)
        metric = metric_cls()
        result = metric._compute_column(real_numeric, synth_numeric)
        assert isinstance(result, float)
        assert np.isfinite(result)

    @pytest.mark.parametrize("metric_id", MARGINAL_TAIL_METRICS)
    def test_metric_nonnegative(self, registry, metric_id, real_numeric, synth_numeric):
        metric_cls = registry.get(metric_id)
        metric = metric_cls()
        result = metric._compute_column(real_numeric, synth_numeric)
        assert result >= 0.0

    @pytest.mark.parametrize("metric_id", MARGINAL_TAIL_METRICS)
    def test_identical_data_low_distance(self, registry, metric_id, identical_numeric):
        real, synth = identical_numeric
        metric_cls = registry.get(metric_id)
        metric = metric_cls()
        result = metric._compute_column(real, synth)
        # Identical data should have zero or near-zero distance
        assert result < 0.01


class TestMarginalScaleMetrics:
    """Test marginal scale metrics."""

    @pytest.mark.parametrize("metric_id", MARGINAL_SCALE_METRICS)
    def test_metric_returns_float(self, registry, metric_id, real_numeric, synth_numeric):
        metric_cls = registry.get(metric_id)
        metric = metric_cls()
        result = metric._compute_column(real_numeric, synth_numeric)
        assert isinstance(result, float)
        assert np.isfinite(result)

    @pytest.mark.parametrize("metric_id", MARGINAL_SCALE_METRICS)
    def test_metric_nonnegative(self, registry, metric_id, real_numeric, synth_numeric):
        metric_cls = registry.get(metric_id)
        metric = metric_cls()
        result = metric._compute_column(real_numeric, synth_numeric)
        assert result >= 0.0

    @pytest.mark.parametrize("metric_id", MARGINAL_SCALE_METRICS)
    def test_identical_data_zero(self, registry, metric_id, identical_numeric):
        real, synth = identical_numeric
        metric_cls = registry.get(metric_id)
        metric = metric_cls()
        result = metric._compute_column(real, synth)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestAllRegisteredMetrics:
    """General tests that apply to all registered metrics."""

    def test_all_metrics_have_name_attribute(self, registry):
        for metric_id in registry.list_ids():
            metric_cls = registry.get(metric_id)
            metric = metric_cls()
            assert hasattr(metric, "name"), f"{metric_id} missing 'name' attribute"

    def test_all_metrics_have_is_distance_attribute(self, registry):
        for metric_id in registry.list_ids("fidelity"):
            metric_cls = registry.get(metric_id)
            metric = metric_cls()
            assert hasattr(metric, "is_distance"), f"{metric_id} missing 'is_distance' attribute"

    def test_fidelity_metrics_registered(self, registry):
        fidelity_ids = registry.list_ids("fidelity")
        # Should have at least the 26 fidelity metrics
        assert len(fidelity_ids) >= 26

    def test_privacy_metrics_registered(self, registry):
        privacy_ids = registry.list_ids("privacy")
        assert len(privacy_ids) >= 8

    def test_utility_metrics_registered(self, registry):
        utility_ids = registry.list_ids("utility")
        assert len(utility_ids) >= 2
