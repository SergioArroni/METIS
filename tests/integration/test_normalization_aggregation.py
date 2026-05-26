"""Integration test: normalization + aggregation chain.

Validates that the full scoring pipeline (raw metric → normalize → aggregate)
produces coherent results for known input distributions.
"""

import numpy as np
import pytest

from metis.shared.aggregation_registry import get_aggregation_function
from metis.shared.normalization import normalize_metric_value


class TestNormalizationAggregationChain:
    """Test that normalization feeds correctly into aggregation."""

    def test_perfect_synthetic_data_scores_high(self):
        """Zero distance metrics → normalized = 1.0 → aggregate near 1.0."""
        metric_ids = ["fidelity.ks", "fidelity.wasserstein", "fidelity.hellinger"]
        raw_values = [0.0, 0.0, 0.0]  # Perfect match

        normalized = [normalize_metric_value(mid, val) for mid, val in zip(metric_ids, raw_values)]
        assert all(v == 1.0 for v in normalized)

        agg_fn = get_aggregation_function("mean")
        score = agg_fn(normalized)
        assert score == pytest.approx(1.0)

    def test_poor_synthetic_data_scores_low(self):
        """High distance metrics → normalized near 0 → aggregate near 0."""
        metric_ids = ["fidelity.ks", "fidelity.hellinger", "fidelity.tvd"]
        raw_values = [0.95, 0.9, 0.85]  # Very different distributions

        normalized = [normalize_metric_value(mid, val) for mid, val in zip(metric_ids, raw_values)]
        # All should be low (1 - value for bounded distance)
        assert all(v < 0.2 for v in normalized)

        agg_fn = get_aggregation_function("mean")
        score = agg_fn(normalized)
        assert score < 0.2

    def test_mixed_quality_intermediate_score(self):
        """Mix of good and bad metrics → intermediate aggregate."""
        pairs = [
            ("fidelity.ks", 0.1),  # Good (low distance)
            ("fidelity.hellinger", 0.8),  # Bad (high distance)
            ("fidelity.tvd", 0.05),  # Good
        ]

        normalized = [normalize_metric_value(mid, val) for mid, val in pairs]
        agg_fn = get_aggregation_function("mean")
        score = agg_fn(normalized)
        assert 0.3 < score < 0.8

    def test_ssd_penalizes_variance_more_than_mean(self):
        """SSD should give lower scores than mean for unbalanced inputs."""
        # Unbalanced: one very good, one very bad
        values = [0.95, 0.1, 0.95]

        mean_fn = get_aggregation_function("mean")
        ssd_fn = get_aggregation_function("ssd")

        mean_score = mean_fn(values)
        ssd_score = ssd_fn(values)

        # SSD should penalize the low outlier more
        assert ssd_score < mean_score

    def test_all_aggregators_handle_normalized_input(self):
        """All aggregators should work on normalized [0,1] values without error."""
        rng = np.random.default_rng(42)
        normalized_values = list(rng.uniform(0, 1, size=10))

        for name in [
            "mean",
            "median",
            "ssd",
            "fsd",
            "trimmed_mean_10",
            "geometric_mean",
            "harmonic_mean",
        ]:
            fn = get_aggregation_function(name)
            result = fn(normalized_values)
            assert 0.0 <= result <= 1.0, f"{name} returned {result}"
