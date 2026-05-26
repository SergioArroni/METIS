"""Tests for metis.application.pipeline.evaluator — metric execution engine.

WHY: The evaluator is the bridge between the plan and actual metric computation.
It must:
  - Correctly dispatch metrics from the registry
  - Isolate failures (one metric crash doesn't kill the run)
  - Route data correctly (cat/num/full based on metric family/requirements)
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from metis.application.pipeline.evaluator import MetricEvaluator
from metis.domain.entities import DatasetSpec, EvalPlan, MetricResult, TransformedData


@pytest.fixture
def mock_registry():
    """Registry that returns controllable mock metrics."""
    registry = MagicMock()

    class FakeMetric:
        family = "fidelity"
        requires_data = "both"

        def __init__(self):
            pass

        def fit(self, real, synth, ctx):
            pass

        def compute(self):
            return MetricResult(
                id="fidelity.ks", value=0.85, details={"p_value": 0.3}, family="fidelity"
            )

    registry.get.return_value = FakeMetric
    return registry


@pytest.fixture
def transformed_data():
    """Minimal TransformedData for testing routing."""
    cat = pd.DataFrame({"gender": ["M", "F", "M"]})
    num = pd.DataFrame({"age": [25.0, 30.0, 35.0], "income": [50000.0, 60000.0, 70000.0]})
    full = pd.concat([cat, num], axis=1)
    return TransformedData(
        cat=cat,
        num=num,
        full=full,
        meta={"gender": {"type": "categorical"}, "age": {"type": "continuous"}},
        excluded_ids=[],
    )


@pytest.fixture
def dataset_spec():
    return DatasetSpec(target="income", task_type="regression")


# =============================================================================
# Basic execution
# =============================================================================


class TestMetricEvaluatorExecution:
    def test_evaluates_all_metrics(self, mock_registry, transformed_data, dataset_spec):
        evaluator = MetricEvaluator(metric_registry=mock_registry)
        plan = EvalPlan(metric_ids=["fidelity.ks", "fidelity.wasserstein"])

        results = evaluator.evaluate(plan, transformed_data, transformed_data, dataset_spec, 42)
        assert len(results) == 2
        assert all(isinstance(r, MetricResult) for r in results)

    def test_failed_metric_produces_error_result(self, transformed_data, dataset_spec):
        """A metric that throws should produce MetricResult with error, not crash."""
        registry = MagicMock()

        class CrashingMetric:
            family = "fidelity"
            requires_data = "both"

            def __init__(self):
                pass

            def fit(self, real, synth, ctx):
                pass

            def compute(self):
                raise RuntimeError("GPU out of memory")

        registry.get.return_value = CrashingMetric
        evaluator = MetricEvaluator(metric_registry=registry)
        plan = EvalPlan(metric_ids=["fidelity.mmd"])

        results = evaluator.evaluate(plan, transformed_data, transformed_data, dataset_spec, 42)
        assert len(results) == 1
        assert "error" in results[0].details
        assert "GPU out of memory" in results[0].details["error"]
        assert results[0].details["error_type"] == "RuntimeError"


# =============================================================================
# Data routing
# =============================================================================


class TestDataRouting:
    """_route_data must select the correct DataFrame slice based on metric metadata."""

    def test_privacy_uses_full(self, transformed_data):
        class PrivacyMetric:
            family = "privacy"
            requires_data = "both"

        real, synth = MetricEvaluator._route_data(PrivacyMetric, transformed_data, transformed_data)
        assert list(real.columns) == list(transformed_data.full.columns)

    def test_utility_uses_full(self, transformed_data):
        class UtilityMetric:
            family = "utility"
            requires_data = "both"

        real, synth = MetricEvaluator._route_data(UtilityMetric, transformed_data, transformed_data)
        assert list(real.columns) == list(transformed_data.full.columns)

    def test_cat_requirement(self, transformed_data):
        class CatMetric:
            family = "fidelity"
            requires_data = "cat"

        real, synth = MetricEvaluator._route_data(CatMetric, transformed_data, transformed_data)
        assert list(real.columns) == list(transformed_data.cat.columns)

    def test_num_requirement(self, transformed_data):
        class NumMetric:
            family = "fidelity"
            requires_data = "num"

        real, synth = MetricEvaluator._route_data(NumMetric, transformed_data, transformed_data)
        assert list(real.columns) == list(transformed_data.num.columns)

    def test_default_uses_full(self, transformed_data):
        class DefaultMetric:
            family = "fidelity"
            requires_data = "both"

        real, synth = MetricEvaluator._route_data(DefaultMetric, transformed_data, transformed_data)
        assert list(real.columns) == list(transformed_data.full.columns)
