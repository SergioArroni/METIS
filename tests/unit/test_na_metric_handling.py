import math

import pandas as pd
import pytest

from metis.application.aggregator import Aggregator
from metis.domain.entities import EvalPlan, MetricResult
from metis.infrastructure.metrics.fidelity.conditional.num_cat.point_biserial import (
    PointBiserialMetric,
)
from metis.shared.normalization import normalize_values_batch


def test_point_biserial_returns_nan_when_no_binary_pairs() -> None:
    real = pd.DataFrame(
        {
            "value": list(range(30)),
            "group": (["A", "B", "C"] * 10),
        }
    )
    synth = real.copy()

    metric = PointBiserialMetric()
    metric.fit(real, synth, {})
    result = metric.compute()

    assert math.isnan(result.value)
    assert result.status == "not_applicable"
    assert result.details["_summary"]["total_pairs"] == 0


def test_normalize_values_batch_preserves_nan_inputs() -> None:
    normalized = normalize_values_batch(
        {"non_applicable": float("nan"), "valid": 0.75},
        "fidelity.point_biserial",
    )

    assert math.isnan(normalized["non_applicable"])
    # Pairwise conditional metrics normalise a delta |real - synth| as a
    # BOUNDED_DISTANCE (since ea9625b): a delta of 0.75 scores 1 - 0.75.
    assert normalized["valid"] == pytest.approx(0.25)


def test_aggregator_skips_nan_metric_results() -> None:
    aggregator = Aggregator()
    results = [
        MetricResult(
            id="fidelity.point_biserial",
            value=float("nan"),
            details={"_summary": {"not_applicable": True}},
            family="fidelity",
            purpose_tags=set(),
        ),
        MetricResult(
            id="fidelity.eta_squared",
            value=0.8,
            details={},
            family="fidelity",
            purpose_tags=set(),
        ),
        MetricResult(
            id="utility.tts",
            value=0.6,
            details={},
            family="utility",
            purpose_tags=set(),
        ),
    ]

    aggregates = aggregator.aggregate(
        results,
        EvalPlan(metric_ids=[result.id for result in results]),
    )

    assert aggregates["fidelity_score"] == pytest.approx(0.8)
    assert aggregates["successful_metrics"] == 2
    assert aggregates["failed_metrics"] == 1


def test_aggregator_nan_family_when_all_metrics_fail() -> None:
    """When every metric in a family is NaN/errored, the family score is NaN
    and it is excluded from the composite."""
    aggregator = Aggregator()
    results = [
        MetricResult(
            id="fidelity.ks",
            value=float("nan"),
            details={"error": "something broke"},
            family="fidelity",
            purpose_tags=set(),
        ),
        MetricResult(
            id="fidelity.wasserstein",
            value=float("nan"),
            details={"error": "also broke"},
            family="fidelity",
            purpose_tags=set(),
        ),
        MetricResult(
            id="utility.tts",
            value=0.7,
            details={},
            family="utility",
            purpose_tags=set(),
        ),
    ]

    aggregates = aggregator.aggregate(
        results,
        EvalPlan(metric_ids=[r.id for r in results]),
    )

    assert math.isnan(aggregates["fidelity_score"])
    # Composite should only include utility (the only finite family)
    assert "fidelity" not in aggregates.get("composite_families", [])
    assert aggregates["composite_score"] == pytest.approx(0.7)


def test_evaluator_error_result_is_nan() -> None:
    """MetricResult returned by the evaluator on exception must have NaN value."""
    from metis.application.pipeline.evaluator import MetricEvaluator
    from metis.domain.entities import DatasetSpec, EvalPlan, TransformedData

    real_df = pd.DataFrame({"a": [1, 2, 3]})
    td = TransformedData(
        cat=pd.DataFrame(),
        num=real_df,
        full=real_df,
        meta={},
        excluded_ids=[],
    )

    evaluator = MetricEvaluator()
    spec = DatasetSpec(target="a", task_type="regression")
    results = evaluator.evaluate(
        plan=EvalPlan(metric_ids=["fidelity.nonexistent_metric"]),
        real=td,
        synth=td,
        spec=spec,
        seed=0,
    )
    assert len(results) == 1
    assert math.isnan(results[0].value)
    assert "error" in results[0].details
