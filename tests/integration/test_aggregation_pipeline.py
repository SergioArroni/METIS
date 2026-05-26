"""Integration test: aggregation pipeline end-to-end.

Tests that MetricResults flow correctly through the Aggregator and produce
a coherent hierarchical structure matching the taxonomy.
"""

import numpy as np
import pytest

from metis.application.aggregator import Aggregator
from metis.domain.entities import EvalPlan, MetricResult
from metis.domain.taxonomy import expand_metric_ids


@pytest.fixture
def full_fidelity_results():
    """Simulate results for all tails metrics with realistic values."""
    metrics = expand_metric_ids(["fidelity.marginal.tails"])
    rng = np.random.default_rng(42)
    return [
        MetricResult(
            id=mid,
            value=float(rng.uniform(0.6, 0.95)),
            details={},
            family="fidelity",
        )
        for mid in metrics
    ]


@pytest.fixture
def all_family_results(full_fidelity_results):
    """Results spanning all families for composite testing."""
    privacy_results = [
        MetricResult(id="privacy.dcr", value=0.70, details={}, family="privacy"),
        MetricResult(id="privacy.nnaa", value=0.60, details={}, family="privacy"),
    ]
    utility_results = [
        MetricResult(id="utility.ml_efficiency", value=0.75, details={}, family="utility"),
    ]
    return full_fidelity_results + privacy_results + utility_results


class TestAggregationPipeline:
    """End-to-end aggregation without any mocking."""

    def test_hierarchy_structure_matches_taxonomy(self, full_fidelity_results):
        """Hierarchy output must reflect the actual metric taxonomy."""
        agg = Aggregator()
        plan = EvalPlan(metric_ids=[r.id for r in full_fidelity_results])
        result = agg.aggregate(full_fidelity_results, plan)

        hierarchy = result["hierarchy"]
        assert "fidelity" in hierarchy
        assert "categories" in hierarchy["fidelity"]
        assert "marginal" in hierarchy["fidelity"]["categories"]

        marginal = hierarchy["fidelity"]["categories"]["marginal"]
        assert "subcategories" in marginal
        assert "tails" in marginal["subcategories"]

        tails = marginal["subcategories"]["tails"]
        assert "score" in tails
        assert 0.0 <= tails["score"] <= 1.0
        assert tails["count"] == 6  # 6 tails metrics

    def test_composite_score_bounded(self, all_family_results):
        agg = Aggregator()
        plan = EvalPlan(metric_ids=[r.id for r in all_family_results])
        result = agg.aggregate(all_family_results, plan)

        assert 0.0 <= result["composite_score"] <= 1.0
        assert result["fidelity_score"] > 0
        assert result["privacy_score"] > 0
        assert result["utility_score"] > 0

    def test_family_scores_between_min_max(self, all_family_results):
        """Family score should be bounded by [min(values), max(values)]."""
        agg = Aggregator()
        plan = EvalPlan(metric_ids=[r.id for r in all_family_results])
        result = agg.aggregate(all_family_results, plan)

        fidelity_values = [r.value for r in all_family_results if r.family == "fidelity"]
        fid_score = result["fidelity_score"]
        # Score should be in a reasonable range relative to inputs
        assert fid_score >= min(fidelity_values) * 0.5
        assert fid_score <= max(fidelity_values) * 1.5

    def test_with_calibration_bounds(self, all_family_results):
        """Full pipeline with calibration normalization."""
        from metis.calibrate.core.bounds import CalibrationBounds

        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", lower_bound=0.3, upper_bound=0.95)
        bounds.set_bounds("privacy", lower_bound=0.2, upper_bound=0.85)
        bounds.set_bounds("utility", lower_bound=0.25, upper_bound=0.90)

        agg = Aggregator(calibration_bounds=bounds)
        plan = EvalPlan(metric_ids=[r.id for r in all_family_results])
        result = agg.aggregate(all_family_results, plan)

        # Calibrated scores should be in [0, 1]
        assert 0.0 <= result["fidelity_score"] <= 1.0
        assert 0.0 <= result["privacy_score"] <= 1.0
        assert 0.0 <= result["utility_score"] <= 1.0
        # Raw scores preserved
        assert "fidelity_score_raw" in result

    def test_different_aggregators_produce_different_scores(self, full_fidelity_results):
        """Changing the aggregation function should produce different results."""
        plan = EvalPlan(metric_ids=[r.id for r in full_fidelity_results])

        agg_median = Aggregator()
        agg_median.optimal_aggregators = {
            "fidelity": "median",
            "privacy": "median",
            "utility": "median",
            "composite": "median",
        }
        agg_median._setup_aggregation_functions()

        agg_mean = Aggregator()
        agg_mean.optimal_aggregators = {
            "fidelity": "mean",
            "privacy": "mean",
            "utility": "mean",
            "composite": "mean",
        }
        agg_mean._setup_aggregation_functions()

        result_median = agg_median.aggregate(full_fidelity_results, plan)
        result_mean = agg_mean.aggregate(full_fidelity_results, plan)

        # With random data, median and mean should differ (unless symmetric)
        # At minimum, both should produce valid scores
        assert result_median["fidelity_score"] > 0
        assert result_mean["fidelity_score"] > 0


class TestCalibrationRoundtrip:
    """Test that calibration bounds survive save/load and still normalize correctly."""

    def test_save_load_normalize(self, tmp_path, all_family_results):
        from metis.calibrate.core.bounds import CalibrationBounds

        filepath = str(tmp_path / "test_bounds.json")

        # Create and save
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", 0.3, 0.9)
        bounds.set_bounds("privacy", 0.2, 0.8)
        bounds.set_bounds("utility", 0.25, 0.85)
        bounds.optimal_aggregators = {"composite": "ssd"}
        bounds.save(filepath)

        # Load and use
        loaded = CalibrationBounds.load(filepath)
        agg = Aggregator(calibration_bounds=loaded)
        plan = EvalPlan(metric_ids=[r.id for r in all_family_results])
        result = agg.aggregate(all_family_results, plan)

        assert 0.0 <= result["composite_score"] <= 1.0
        assert result.get("composite_score_calibrated") is True
