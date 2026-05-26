"""Tests for metis.domain.entities — core value objects.

WHY: These frozen dataclasses define the contracts between all layers.
A broken entity validation or accessor silently corrupts downstream scoring.
"""

import math

import pytest

from metis.domain.entities import DatasetSpec, EvalPlan, MetricResult, ReportSpec

# =============================================================================
# DatasetSpec
# =============================================================================


class TestDatasetSpec:
    """DatasetSpec is the cornerstone of data description.

    Risk: incorrect validation could accept invalid configs silently.
    """

    def test_minimal_creation(self):
        spec = DatasetSpec()
        assert spec.target is None
        assert spec.task_type is None
        assert spec.dtypes == {}
        assert spec.constraints == {}

    def test_single_target(self):
        spec = DatasetSpec(target="income", task_type="regression")
        assert spec.target == "income"
        assert spec.target_list == ["income"]

    def test_multi_target(self):
        spec = DatasetSpec(target=["col_a", "col_b"])
        assert spec.target_list == ["col_a", "col_b"]

    def test_none_target_gives_empty_list(self):
        spec = DatasetSpec(target=None)
        assert spec.target_list == []

    def test_frozen_immutability(self):
        spec = DatasetSpec(target="x")
        with pytest.raises(AttributeError):
            spec.target = "y"  # type: ignore[misc]

    def test_empty_target_list_rejected(self):
        with pytest.raises(ValueError, match="Target list cannot be empty"):
            DatasetSpec(target=[])

    def test_non_string_targets_rejected(self):
        with pytest.raises(ValueError, match="All targets must be string"):
            DatasetSpec(target=["valid", 123])  # type: ignore[list-item]

    def test_invalid_task_type_rejected(self):
        with pytest.raises(ValueError, match="task_type must be"):
            DatasetSpec(task_type="clustering")  # type: ignore[arg-type]

    def test_invalid_target_type_rejected(self):
        with pytest.raises(ValueError, match="Target must be a string"):
            DatasetSpec(target=42)  # type: ignore[arg-type]


# =============================================================================
# EvalPlan
# =============================================================================


class TestEvalPlan:
    """EvalPlan drives the entire metric selection.

    Risk: accepting empty metric lists or invalid seeds silently breaks runs.
    """

    def test_valid_creation(self):
        plan = EvalPlan(metric_ids=["fidelity.ks"], seed=0, cv_splits=5)
        assert plan.metric_ids == ["fidelity.ks"]
        assert plan.seed == 0
        assert plan.cv_splits == 5

    def test_defaults(self):
        plan = EvalPlan(metric_ids=["fidelity.ks"])
        assert plan.seed == 42
        assert plan.cv_splits == 3

    def test_empty_metric_ids_rejected(self):
        with pytest.raises(ValueError, match="At least one metric_id"):
            EvalPlan(metric_ids=[])

    def test_negative_seed_rejected(self):
        with pytest.raises(ValueError, match="Seed must be non-negative"):
            EvalPlan(metric_ids=["fidelity.ks"], seed=-1)

    def test_cv_splits_minimum(self):
        with pytest.raises(ValueError, match="CV splits must be at least 2"):
            EvalPlan(metric_ids=["fidelity.ks"], cv_splits=1)

    def test_frozen(self):
        plan = EvalPlan(metric_ids=["fidelity.ks"])
        with pytest.raises(AttributeError):
            plan.seed = 99  # type: ignore[misc]


# =============================================================================
# MetricResult
# =============================================================================


class TestMetricResult:
    """MetricResult carries individual metric scores through the pipeline.

    Risk: invalid family values or empty IDs could corrupt aggregation.
    """

    def test_valid_creation(self):
        r = MetricResult(id="fidelity.ks", value=0.85, details={}, family="fidelity")
        assert r.id == "fidelity.ks"
        assert r.value == 0.85
        assert r.family == "fidelity"
        assert r.purpose_tags == set()

    def test_nan_value_allowed(self):
        """NaN signals a failed metric — must not be rejected at construction."""
        r = MetricResult(id="fidelity.mmd", value=float("nan"), details={}, family="fidelity")
        assert math.isnan(r.value)

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="Metric ID cannot be empty"):
            MetricResult(id="", value=0.5, details={}, family="fidelity")

    def test_invalid_family_rejected(self):
        with pytest.raises(ValueError, match="Invalid family"):
            MetricResult(id="x.y", value=0.5, details={}, family="invalid")

    def test_error_in_details(self):
        """Convention: 'error' key in details marks a failed metric."""
        r = MetricResult(
            id="fidelity.mmd",
            value=float("nan"),
            details={"error": "timeout"},
            family="fidelity",
        )
        assert "error" in r.details

    def test_with_purpose_tags(self):
        r = MetricResult(
            id="privacy.dcr",
            value=0.7,
            details={},
            family="privacy",
            purpose_tags={"distance", "empirical"},
        )
        assert "distance" in r.purpose_tags


# =============================================================================
# RunSummary
# =============================================================================


class TestRunSummary:
    """RunSummary aggregates a full evaluation run.

    Risk: get_results_by_family or get_family_score returning wrong data
    means incorrect reporting.
    """

    def test_get_results_by_family(self, sample_run_summary):
        fidelity = sample_run_summary.get_results_by_family("fidelity")
        assert len(fidelity) == 3
        assert all(r.family == "fidelity" for r in fidelity)

    def test_get_results_by_family_empty(self, sample_run_summary):
        unknown = sample_run_summary.get_results_by_family("nonexistent")
        assert unknown == []

    def test_get_family_score(self, sample_run_summary):
        assert sample_run_summary.get_family_score("fidelity") == 0.82
        assert sample_run_summary.get_family_score("privacy") == 0.60

    def test_get_family_score_missing(self, sample_run_summary):
        assert sample_run_summary.get_family_score("nonexistent") == 0.0


# =============================================================================
# ReportSpec
# =============================================================================


class TestReportSpec:
    def test_valid_creation(self):
        spec = ReportSpec(formats=["json", "markdown"], output_dir="reports/")
        assert spec.formats == ["json", "markdown"]
        assert spec.include_details is True

    def test_empty_formats_rejected(self):
        with pytest.raises(ValueError, match="At least one format"):
            ReportSpec(formats=[], output_dir="reports/")

    def test_empty_output_dir_rejected(self):
        with pytest.raises(ValueError, match="Output directory cannot be empty"):
            ReportSpec(formats=["json"], output_dir="")
