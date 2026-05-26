"""
Result data structure for scales metrics aggregation.
"""

from dataclasses import dataclass
from typing import Any

from metis.domain.entities import MetricResult
from metis.shared.aggregation import AggregationResult


@dataclass
class ScalesResult(AggregationResult):
    """
    Result of scales aggregation.

    Extends the base AggregationResult with scales-specific functionality.

    Attributes:
        score: Final aggregated score Q ∈ [0, 1]
        column_scores: μᵢ for each column
        metric_details: Full details per metric
        metrics_used: list of metric names used
        n_columns: Number of columns processed
        n_metrics: Number of metrics computed
    """

    def to_metric_result(self) -> MetricResult:
        """
        Convert to domain MetricResult for reporting integration.

        Returns:
            MetricResult compatible with the reporting infrastructure
        """
        details: dict[str, Any] = {
            "column_scores": self.column_scores,
            "metrics_used": self.metrics_used,
            "n_columns": self.n_columns,
            "n_metrics": self.n_metrics,
            "per_metric_summary": self.get_per_metric_summary(),
        }

        return MetricResult(
            id="fidelity.marginal.scales",
            value=self.score,
            details=details,
            family="fidelity",
            purpose_tags={"fidelity", "marginal", "scales", "distribution"},
        )

    def get_report_data(self) -> dict[str, Any]:
        """Get structured data for report generation."""
        return super().get_report_data(metric_id="fidelity.marginal.scales", category="escala")

    @classmethod
    def empty(cls, metrics_used: list[str]) -> "ScalesResult":
        """
        Create an empty result when no data is available.

        Args:
            metrics_used: list of metric names

        Returns:
            Empty ScalesResult instance
        """
        return cls(
            score=0.0,
            column_scores={},
            metric_details={},
            metrics_used=metrics_used,
            n_columns=0,
            n_metrics=len(metrics_used),
        )
