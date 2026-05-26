"""Unit-test specific fixtures with mocked dependencies."""

import pytest

from metis.domain.entities import MetricResult


@pytest.fixture
def failed_metric_result() -> MetricResult:
    """A MetricResult representing a failed computation."""
    return MetricResult(
        id="fidelity.mmd",
        value=float("nan"),
        details={"error": "Division by zero", "error_type": "ZeroDivisionError"},
        family="fidelity",
    )


@pytest.fixture
def mixed_metric_results(sample_metric_results, failed_metric_result) -> list[MetricResult]:
    """Results mixing successful and failed metrics."""
    return sample_metric_results + [failed_metric_result]
