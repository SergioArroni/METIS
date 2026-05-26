"""Tests for metis.shared.distributions — probability distribution utilities.

WHY: These functions feed all marginal metrics (KS, TVD, JS, etc.).
Incorrect distribution alignment means all downstream metrics are wrong.
Key invariants: distributions must sum to 1, have same length after alignment.
"""

import numpy as np
import pandas as pd
import pytest

from metis.shared.distributions import (
    align_distributions,
    compute_empirical_cdf,
    compute_histogram,
    get_distribution,
)

# =============================================================================
# get_distribution
# =============================================================================


class TestGetDistribution:
    """Risk: distribution not summing to 1 → invalid metric computations."""

    def test_categorical_series(self):
        s = pd.Series(["A", "A", "B", "C", "C", "C"])
        dist = get_distribution(s)
        assert dist.sum() == pytest.approx(1.0)
        assert len(dist) == 3  # 3 unique categories

    def test_numeric_series(self):
        s = pd.Series(np.random.default_rng(42).uniform(0, 100, size=200))
        dist = get_distribution(s, n_bins=20)
        # Histogram-based, approximately sums to 1 (density normalization)
        # Note: density=True means area=1 but sum of bins != 1 unless divided
        assert len(dist) == 20
        # After internal normalization: sum should be approx 1
        assert dist.sum() == pytest.approx(1.0, abs=0.1)

    def test_low_cardinality_numeric_treated_as_categorical(self):
        """Numeric with < 20 unique values → treated as categorical."""
        s = pd.Series([1, 1, 2, 2, 3, 3, 3])
        dist = get_distribution(s)
        assert dist.sum() == pytest.approx(1.0)

    def test_single_value_series(self):
        s = pd.Series(["X", "X", "X"])
        dist = get_distribution(s)
        assert len(dist) == 1
        assert dist[0] == pytest.approx(1.0)


# =============================================================================
# align_distributions
# =============================================================================


class TestAlignDistributions:
    """Risk: misaligned distributions → comparing apples to oranges."""

    def test_same_categories(self):
        real = pd.Series(["A", "B", "C", "A", "B"])
        synth = pd.Series(["A", "B", "C", "C", "B"])
        p, q = align_distributions(real, synth)
        assert len(p) == len(q)
        assert p.sum() == pytest.approx(1.0)
        assert q.sum() == pytest.approx(1.0)

    def test_disjoint_categories(self):
        """Real has categories not in synth and vice versa."""
        real = pd.Series(["A", "B", "A"])
        synth = pd.Series(["B", "C", "C"])
        p, q = align_distributions(real, synth)
        assert len(p) == len(q)
        assert len(p) == 3  # A, B, C
        # All values > 0 (epsilon added)
        assert np.all(p > 0)
        assert np.all(q > 0)

    def test_numeric_alignment(self):
        rng = np.random.default_rng(42)
        real = pd.Series(rng.normal(0, 1, size=100))
        synth = pd.Series(rng.normal(0.5, 1, size=100))
        p, q = align_distributions(real, synth, n_bins=20)
        assert len(p) == len(q) == 20
        assert p.sum() == pytest.approx(1.0)
        assert q.sum() == pytest.approx(1.0)

    def test_epsilon_prevents_zeros(self):
        """No zeros in output — critical for KL divergence."""
        real = pd.Series(["A", "A", "A"])
        synth = pd.Series(["B", "B", "B"])
        p, q = align_distributions(real, synth)
        assert np.all(p > 0)
        assert np.all(q > 0)

    def test_nan_handling(self):
        """NaN values should be dropped, not corrupt the distribution."""
        real = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0])
        synth = pd.Series([1.5, 2.5, 3.5, np.nan, 4.5])
        p, q = align_distributions(real, synth, n_bins=5)
        assert len(p) == len(q)
        assert p.sum() == pytest.approx(1.0)


# =============================================================================
# compute_histogram & compute_empirical_cdf
# =============================================================================


class TestComputeHistogram:
    def test_basic(self):
        s = pd.Series(np.linspace(0, 1, 100))
        hist, edges = compute_histogram(s, bins=10)
        assert len(hist) == 10
        assert len(edges) == 11

    def test_nan_dropped(self):
        s = pd.Series([1.0, 2.0, np.nan, 3.0])
        hist, edges = compute_histogram(s, bins=3, density=False)
        assert hist.sum() == 3  # Only 3 valid values


class TestComputeEmpiricalCDF:
    def test_basic(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        values, cdf = compute_empirical_cdf(s)
        assert values[0] == 1.0
        assert values[-1] == 5.0
        assert cdf[-1] == 1.0
        assert cdf[0] == pytest.approx(0.2)

    def test_sorted_output(self):
        s = pd.Series([5.0, 1.0, 3.0, 2.0, 4.0])
        values, cdf = compute_empirical_cdf(s)
        assert list(values) == sorted(values)

    def test_nan_dropped(self):
        s = pd.Series([1.0, np.nan, 3.0])
        values, cdf = compute_empirical_cdf(s)
        assert len(values) == 2
        assert cdf[-1] == 1.0
