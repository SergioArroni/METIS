"""Tests for metis.domain.taxonomy — metric catalog and expansion.

WHY: expand_metric_ids() is the gateway that converts user shorthand
into concrete metric IDs. A missed expansion = silently missing metrics.
group_metrics_by_hierarchy() feeds the hierarchical scoring logic.
"""

import pytest

from metis.domain.taxonomy import (
    FAMILIES,
    FIDELITY_METRICS,
    PRIVACY_METRICS,
    UTILITY_METRICS,
    expand_metric_ids,
    get_metric_hierarchy,
    group_metrics_by_hierarchy,
)

# =============================================================================
# FAMILIES constant
# =============================================================================


class TestFamilies:
    def test_contains_all_three(self):
        assert {"fidelity", "utility", "privacy"} == FAMILIES


# =============================================================================
# expand_metric_ids
# =============================================================================


class TestExpandMetricIds:
    """Risk: partial expansion silently drops metrics from evaluation."""

    def test_concrete_id_passes_through(self):
        result = expand_metric_ids(["fidelity.ks"])
        assert result == ["fidelity.ks"]

    def test_family_shorthand_expands_all(self):
        result = expand_metric_ids(["fidelity"])
        # Must contain all fidelity metrics (31 total)
        assert "fidelity.ks" in result
        assert "fidelity.pearson" in result
        assert "fidelity.outliers_coverage" in result
        assert len(result) >= 30  # at least 30+ fidelity metrics

    def test_category_shorthand(self):
        result = expand_metric_ids(["fidelity.marginal"])
        # marginal = tails(6) + scales(5) + coverage(6) = 17
        assert len(result) == 17
        assert "fidelity.ks" in result
        assert "fidelity.delta_mean" in result
        assert "fidelity.tvd" in result

    def test_subcategory_shorthand(self):
        result = expand_metric_ids(["fidelity.marginal.tails"])
        expected = [
            "fidelity.ks",
            "fidelity.wasserstein",
            "fidelity.anderson_darling",
            "fidelity.hellinger",
            "fidelity.kde_ise",
            "fidelity.delta_exceedance",
        ]
        assert result == expected

    def test_privacy_family_expansion(self):
        result = expand_metric_ids(["privacy"])
        assert "privacy.dcr" in result
        assert "privacy.differential_privacy" in result
        assert len(result) == 9

    def test_utility_family_expansion(self):
        result = expand_metric_ids(["utility"])
        assert "utility.ml_efficiency" in result
        assert "utility.tts" in result
        assert len(result) == 5

    def test_mixed_shorthand_and_concrete(self):
        result = expand_metric_ids(["fidelity.marginal.tails", "privacy.dcr"])
        assert "fidelity.ks" in result
        assert "privacy.dcr" in result
        assert len(result) == 7  # 6 tails + 1 privacy

    def test_deduplication(self):
        """Repeated entries should not produce duplicates."""
        result = expand_metric_ids(["fidelity.ks", "fidelity.marginal.tails"])
        assert result.count("fidelity.ks") == 1

    def test_unknown_entry_passes_through(self):
        """Unknown IDs pass through for downstream registry error."""
        result = expand_metric_ids(["unknown.metric"])
        assert result == ["unknown.metric"]

    def test_empty_input(self):
        assert expand_metric_ids([]) == []


# =============================================================================
# get_metric_hierarchy
# =============================================================================


class TestGetMetricHierarchy:
    def test_known_metric(self):
        h = get_metric_hierarchy("fidelity.ks")
        assert h["family"] == "fidelity"
        assert h["category"] == "marginal"
        assert h["subcategory"] == "tails"

    def test_global_metric_no_subcategory(self):
        h = get_metric_hierarchy("fidelity.correlation_matrix")
        assert h["family"] == "fidelity"
        assert h["category"] == "global"
        assert h["subcategory"] is None

    def test_privacy_metric(self):
        h = get_metric_hierarchy("privacy.dcr")
        assert h["family"] == "privacy"
        assert h["subcategory"] == "empirical_similarity"

    def test_short_name_lookup(self):
        """Should resolve 'ks' → fidelity.ks."""
        h = get_metric_hierarchy("ks")
        assert h["family"] == "fidelity"
        assert h["category"] == "marginal"

    def test_unknown_metric_with_family_prefix(self):
        h = get_metric_hierarchy("fidelity.unknown_metric")
        assert h["family"] == "fidelity"
        assert h["category"] == "unknown"

    def test_completely_unknown(self):
        h = get_metric_hierarchy("totally_unknown")
        assert h["family"] == "unknown"


# =============================================================================
# group_metrics_by_hierarchy
# =============================================================================


class TestGroupMetricsByHierarchy:
    def test_groups_correctly(self):
        ids = ["fidelity.ks", "fidelity.wasserstein", "privacy.dcr"]
        grouped = group_metrics_by_hierarchy(ids)

        assert "fidelity" in grouped
        assert "privacy" in grouped
        assert "tails" in grouped["fidelity"]["marginal"]
        assert "fidelity.ks" in grouped["fidelity"]["marginal"]["tails"]

    def test_direct_category_metrics(self):
        """Global metrics have no subcategory → grouped under '_direct'."""
        ids = ["fidelity.correlation_matrix", "fidelity.mmd"]
        grouped = group_metrics_by_hierarchy(ids)
        assert "_direct" in grouped["fidelity"]["global"]
        assert "fidelity.correlation_matrix" in grouped["fidelity"]["global"]["_direct"]

    def test_empty_input(self):
        assert group_metrics_by_hierarchy([]) == {}

    @pytest.mark.parametrize(
        "family,taxonomy",
        [
            ("fidelity", FIDELITY_METRICS),
            ("privacy", PRIVACY_METRICS),
            ("utility", UTILITY_METRICS),
        ],
    )
    def test_all_taxonomy_metrics_resolvable(self, family, taxonomy):
        """Every metric in the taxonomy must be resolvable by get_metric_hierarchy."""
        for cat_data in taxonomy.values():
            metrics = []
            if "metrics" in cat_data:
                metrics.extend(cat_data["metrics"])
            if "subcategories" in cat_data:
                for subcat_metrics in cat_data["subcategories"].values():
                    metrics.extend(subcat_metrics)
            for mid in metrics:
                h = get_metric_hierarchy(mid)
                assert h["family"] == family, f"{mid} has wrong family"
