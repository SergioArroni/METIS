"""
Result data structure for tails metrics aggregation.
"""

from dataclasses import dataclass
from typing import Any

from metis.shared.aggregation import AggregationResult


@dataclass
class TailsResult(AggregationResult):
    """
    Result of tails aggregation.

    Extends the base AggregationResult with tails-specific functionality.

    Attributes:
        score: Final aggregated score Q ∈ [0, 1]
        column_scores: μᵢ for each column
        metric_details: Full details per metric
        metrics_used: list of metric names used
        n_columns: Number of columns processed
        n_metrics: Number of metrics computed
    """

    def get_report_data(self) -> dict[str, Any]:
        """Get structured data for report generation."""
        return super().get_report_data(
            metric_id="fidelity.marginal.tails", category="colas de distribución"
        )

    @classmethod
    def empty(cls, metrics_used: list[str]) -> "TailsResult":
        """
        Create an empty result when no data is available.

        Args:
            metrics_used: list of metric names

        Returns:
            Empty TailsResult instance
        """
        return cls(
            score=0.0,
            column_scores={},
            metric_details={},
            metrics_used=metrics_used,
            n_columns=0,
            n_metrics=len(metrics_used),
        )
