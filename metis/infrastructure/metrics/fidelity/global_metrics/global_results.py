"""
Result data structures for global metrics.

Provides common result classes used across all global metric types.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class GlobalMetricResult:
    """
    Result of a global metric computation.

    Global metrics measure dataset-level similarity rather than
    column-level or pair-level comparisons.

    Attributes:
        metric_name: Name of the metric
        raw_value: Original metric value before normalization
        normalized_value: Value in [0, 1] where 1 = best quality
        is_valid: Whether the computation was successful
        details: Additional metric-specific details
        error: Error message if computation failed

    Example:
        >>> result = GlobalMetricResult(
        ...     metric_name="mmd", raw_value=0.05, normalized_value=0.95, is_valid=True
        ... )
        >>> result.is_valid
        True
    """

    metric_name: str
    raw_value: float
    normalized_value: float  # In [0, 1], where 1 = best
    is_valid: bool
    details: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "metric_name": self.metric_name,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "is_valid": self.is_valid,
            "details": self.details,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlobalMetricResult":
        """
        Create from dictionary.

        Args:
            data: Dictionary with result data

        Returns:
            GlobalMetricResult instance
        """
        return cls(
            metric_name=data["metric_name"],
            raw_value=data["raw_value"],
            normalized_value=data["normalized_value"],
            is_valid=data["is_valid"],
            details=data.get("details"),
            error=data.get("error"),
        )

    @classmethod
    def invalid(cls, metric_name: str, error: str) -> "GlobalMetricResult":
        """
        Create an invalid result with error message.

        Args:
            metric_name: Name of the metric
            error: Error description

        Returns:
            Invalid GlobalMetricResult instance
        """
        return cls(
            metric_name=metric_name,
            raw_value=float("nan"),
            normalized_value=0.0,
            is_valid=False,
            error=error,
        )


@dataclass
class GlobalFidelityResult:
    """
    Result of global fidelity aggregation.

    Contains aggregated scores for all global metrics.

    Attributes:
        score: Overall aggregated score in [0, 1]
        outliers_coverage_score: Score for outliers/structural coverage
        correlation_matrix_score: Score for correlation matrix similarity
        mmd_score: Score for Maximum Mean Discrepancy
        energy_distance_score: Score for Energy Distance
        metric_details: Detailed results for each metric
        n_metrics_computed: Number of metrics successfully computed
    """

    score: float

    outliers_coverage_score: float
    correlation_matrix_score: float
    mmd_score: float
    energy_distance_score: float

    metric_details: dict[str, GlobalMetricResult]
    n_metrics_computed: int

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "score": self.score,
            "outliers_coverage_score": self.outliers_coverage_score,
            "correlation_matrix_score": self.correlation_matrix_score,
            "mmd_score": self.mmd_score,
            "energy_distance_score": self.energy_distance_score,
            "metric_details": {k: v.to_dict() for k, v in self.metric_details.items()},
            "n_metrics_computed": self.n_metrics_computed,
        }
