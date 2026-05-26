"""Tests for metis.application.aggregator — the scoring engine.

WHY: The Aggregator class combines individual metric scores into family scores
and a composite index. This is the most complex numeric logic in the pipeline:
  - Hierarchical scoring (metrics → subcategories → categories → families → composite)
  - Multiple aggregation formats (new per-family, old layers, tuned per-family)
  - Calibration normalization
  - NaN propagation control

A bug here silently corrupts ALL evaluation results.
"""

import math
from unittest.mock import MagicMock

import pytest

from metis.application.aggregator import Aggregator
from metis.domain.entities import EvalPlan, MetricResult


@pytest.fixture
def basic_aggregator():
    """Aggregator with defaults (no calibration, no tuning)."""
    return Aggregator()


@pytest.fixture
def eval_plan():
    return EvalPlan(metric_ids=["fidelity.ks", "fidelity.wasserstein"], seed=42)


@pytest.fixture
def fidelity_results():
    """Multiple fidelity results for testing family scoring."""
    return [
        MetricResult(id="fidelity.ks", value=0.85, details={}, family="fidelity"),
        MetricResult(id="fidelity.wasserstein", value=0.72, details={}, family="fidelity"),
        MetricResult(id="fidelity.hellinger", value=0.90, details={}, family="fidelity"),
    ]


@pytest.fixture
def multi_family_results(fidelity_results):
    """Results across all families."""
    return fidelity_results + [
        MetricResult(id="privacy.dcr", value=0.65, details={}, family="privacy"),
        MetricResult(id="privacy.nnaa", value=0.70, details={}, family="privacy"),
        MetricResult(id="utility.ml_efficiency", value=0.80, details={}, family="utility"),
    ]


# =============================================================================
# Initialization
# =============================================================================


class TestAggregatorInit:
    def test_default_config(self, basic_aggregator):
        assert basic_aggregator.risk_aversion == 5.0
        assert basic_aggregator.calibration_bounds is None
        assert basic_aggregator.optimal_aggregators is None

    def test_custom_risk_aversion(self):
        agg = Aggregator(risk_aversion=10.0)
        assert agg.risk_aversion == 10.0

    def test_default_aggregation_functions_set(self, basic_aggregator):
        assert basic_aggregator.agg_layers_1_2_name == "median"
        assert basic_aggregator.agg_layers_3_4_name == "ssd"
        assert basic_aggregator.composite_agg_name == "ssd"

    def test_old_format_aggregators(self):
        agg = Aggregator(optimal_aggregators_path=None)
        # Inject old format manually
        agg.optimal_aggregators = {"layers_1_2": "mean", "layers_3_4": "trimmed_mean_10"}
        agg._setup_aggregation_functions()
        assert agg.agg_layers_1_2_name == "mean"
        assert agg.agg_layers_3_4_name == "trimmed_mean_10"

    def test_new_format_aggregators(self):
        agg = Aggregator()
        agg.optimal_aggregators = {
            "fidelity": {"level_1": "mean", "level_2": "median", "level_3": "ssd"},
            "privacy": {"level_1": "mean", "level_2": "median", "level_3": "ssd"},
            "utility": {"level_1": "mean", "level_2": "mean", "level_3": "ssd"},
            "composite": "trimmed_mean_10",
        }
        agg._setup_aggregation_functions()
        assert agg.composite_agg_name == "trimmed_mean_10"

    def test_per_family_tuned_format(self):
        """Format from tune_from_metrics(): {"fidelity": "median", ...}"""
        agg = Aggregator()
        agg.optimal_aggregators = {
            "fidelity": "median",
            "privacy": "mean",
            "utility": "trimmed_mean_10",
            "composite": "ssd",
        }
        agg._setup_aggregation_functions()
        assert hasattr(agg, "_family_agg_funcs")
        assert "fidelity" in agg._family_agg_funcs


# =============================================================================
# _is_valid_metric_result & _filter_finite_values
# =============================================================================


class TestValidation:
    def test_valid_result(self, basic_aggregator):
        r = MetricResult(id="x", value=0.5, details={}, family="fidelity")
        assert basic_aggregator._is_valid_metric_result(r)

    def test_error_result_invalid(self, basic_aggregator):
        r = MetricResult(id="x", value=0.5, details={"error": "boom"}, family="fidelity")
        assert not basic_aggregator._is_valid_metric_result(r)

    def test_nan_value_invalid(self, basic_aggregator):
        r = MetricResult(id="x", value=float("nan"), details={}, family="fidelity")
        assert not basic_aggregator._is_valid_metric_result(r)

    def test_inf_value_invalid(self, basic_aggregator):
        r = MetricResult(id="x", value=float("inf"), details={}, family="fidelity")
        assert not basic_aggregator._is_valid_metric_result(r)

    def test_filter_finite(self, basic_aggregator):
        values = [0.5, float("nan"), 0.7, float("inf"), 0.3]
        result = basic_aggregator._filter_finite_values(values)
        assert result == [0.5, 0.7, 0.3]


# =============================================================================
# _calculate_family_score
# =============================================================================


class TestCalculateFamilyScore:
    def test_valid_results(self, basic_aggregator, fidelity_results):
        score = basic_aggregator._calculate_family_score(fidelity_results, "fidelity")
        assert 0.0 <= score <= 1.0

    def test_empty_results(self, basic_aggregator):
        score = basic_aggregator._calculate_family_score([], "fidelity")
        assert math.isnan(score)

    def test_all_failed_results(self, basic_aggregator):
        results = [
            MetricResult(id="x", value=float("nan"), details={"error": "fail"}, family="fidelity"),
            MetricResult(id="y", value=float("nan"), details={"error": "fail"}, family="fidelity"),
        ]
        score = basic_aggregator._calculate_family_score(results, "fidelity")
        assert math.isnan(score)

    def test_mixed_valid_and_failed(self, basic_aggregator):
        """Failed metrics should be excluded, not corrupt the score."""
        results = [
            MetricResult(id="a", value=0.8, details={}, family="fidelity"),
            MetricResult(id="b", value=float("nan"), details={"error": "x"}, family="fidelity"),
            MetricResult(id="c", value=0.6, details={}, family="fidelity"),
        ]
        score = basic_aggregator._calculate_family_score(results, "fidelity")
        assert 0.0 <= score <= 1.0
        assert not math.isnan(score)

    def test_uses_family_specific_aggregator(self):
        """When per-family tuned aggregators are set, they should be used."""
        agg = Aggregator()
        agg.optimal_aggregators = {
            "fidelity": "mean",
            "privacy": "mean",
            "utility": "mean",
            "composite": "mean",
        }
        agg._setup_aggregation_functions()

        results = [
            MetricResult(id="a", value=0.4, details={}, family="fidelity"),
            MetricResult(id="b", value=0.6, details={}, family="fidelity"),
            MetricResult(id="c", value=0.8, details={}, family="fidelity"),
        ]
        score = agg._calculate_family_score(results, "fidelity")
        # Mean of [0.4, 0.6, 0.8] = 0.6
        assert score == pytest.approx(0.6)


# =============================================================================
# aggregate() — full flow
# =============================================================================


class TestAggregate:
    def test_produces_composite_score(self, basic_aggregator, multi_family_results, eval_plan):
        result = basic_aggregator.aggregate(multi_family_results, eval_plan)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_produces_family_scores(self, basic_aggregator, multi_family_results, eval_plan):
        result = basic_aggregator.aggregate(multi_family_results, eval_plan)
        assert "fidelity_score" in result
        assert "privacy_score" in result
        assert "utility_score" in result

    def test_counts_metrics(self, basic_aggregator, multi_family_results, eval_plan):
        result = basic_aggregator.aggregate(multi_family_results, eval_plan)
        assert result["total_metrics"] == 6
        assert result["successful_metrics"] == 6
        assert result["failed_metrics"] == 0

    def test_with_failed_metrics(self, basic_aggregator, eval_plan):
        results = [
            MetricResult(id="fidelity.ks", value=0.8, details={}, family="fidelity"),
            MetricResult(
                id="fidelity.mmd",
                value=float("nan"),
                details={"error": "crash"},
                family="fidelity",
            ),
        ]
        agg = basic_aggregator.aggregate(results, eval_plan)
        assert agg["successful_metrics"] == 1
        assert agg["failed_metrics"] == 1

    def test_hierarchy_included(self, basic_aggregator, multi_family_results, eval_plan):
        result = basic_aggregator.aggregate(multi_family_results, eval_plan)
        assert "hierarchy" in result
        assert isinstance(result["hierarchy"], dict)

    def test_empty_results(self, basic_aggregator, eval_plan):
        """Empty results should not crash, just produce zero metrics."""
        result = basic_aggregator.aggregate([], eval_plan)
        assert result["total_metrics"] == 0
        assert result["successful_metrics"] == 0


# =============================================================================
# Calibration normalization
# =============================================================================


class TestCalibrationNormalization:
    def test_with_bounds(self):
        """CalibrationBounds should normalize family scores."""
        mock_bounds = MagicMock()
        mock_bounds.normalize_with_bounds.return_value = 0.75
        mock_bounds.get_all_families.return_value = ["fidelity"]

        agg = Aggregator(calibration_bounds=mock_bounds)
        results = [
            MetricResult(id="fidelity.ks", value=0.8, details={}, family="fidelity"),
        ]
        plan = EvalPlan(metric_ids=["fidelity.ks"])
        output = agg.aggregate(results, plan)

        # Calibrated score should be applied
        assert output["fidelity_score"] == 0.75
        mock_bounds.normalize_with_bounds.assert_called_once()


# =============================================================================
# _hypervolume_aggregator
# =============================================================================


class TestHypervolumeAggregator:
    def test_geometric_mean_approximation(self):
        result = Aggregator._hypervolume_aggregator([0.5, 0.5, 0.5])
        assert result == pytest.approx(0.5)

    def test_with_perfect_scores(self):
        result = Aggregator._hypervolume_aggregator([1.0, 1.0, 1.0])
        assert result == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        result = Aggregator._hypervolume_aggregator([])
        assert result == 0.0

    def test_clips_zeros(self):
        """Zeros should be clipped to epsilon, not crash geometric mean."""
        result = Aggregator._hypervolume_aggregator([0.0, 0.5, 1.0])
        assert 0.0 < result < 0.5  # Non-zero thanks to epsilon clipping
