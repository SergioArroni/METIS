"""Unit tests for metric normalization utilities."""

import pytest

from metis.shared.normalization import (
    METRIC_NORMALIZATION_MAP,
    NormalizationType,
    get_normalizer,
    normalize_metric_value,
)


class TestNormalization:
    def test_utility_scores_use_similarity_family(self):
        assert METRIC_NORMALIZATION_MAP["utility.classification"] == NormalizationType.SIMILARITY
        assert METRIC_NORMALIZATION_MAP["utility.regression"] == NormalizationType.SIMILARITY
        assert METRIC_NORMALIZATION_MAP["utility.ml_efficiency"] == NormalizationType.SIMILARITY

    def test_utility_score_normalization_is_clamped_identity(self):
        normalizer = get_normalizer("utility.classification")

        assert normalizer(0.75) == pytest.approx(0.75)
        assert normalizer(1.2) == pytest.approx(1.0)
        assert normalizer(-0.1) == pytest.approx(0.0)

    def test_pairwise_fidelity_deltas_use_distance_orientation(self):
        similarity_value = normalize_metric_value("fidelity.pearson", 0.18)
        utility_value = normalize_metric_value("utility.classification", 0.82)

        assert similarity_value == pytest.approx(0.82)
        assert utility_value == pytest.approx(0.82)

    def test_pairwise_mutual_information_delta_uses_distance_normalization(self):
        normalized = normalize_metric_value("fidelity.mi", 0.01)

        assert normalized > 0.9
