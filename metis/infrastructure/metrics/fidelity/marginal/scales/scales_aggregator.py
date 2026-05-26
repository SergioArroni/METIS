"""
Scales Aggregator - Orchestrates scale metrics and computes aggregated scores.

Uses Stochastic Dominance (FSD+SSD) for principled aggregation.
Integrates with the reporting infrastructure for output generation.
"""

from metis.shared.aggregation import BaseColumnAggregator
from metis.shared.results import ColumnMetricResult

from ...fidelity_base import NumericColumnMetric as MarginalMetric
from .cohens_d import CohensD
from .delta_iqr import DeltaIQRMetric
from .delta_mad import DeltaMADMetric
from .delta_mean import DeltaMeanMetric
from .delta_median import DeltaMedianMetric
from .scales_result import ScalesResult

# Registry of available scale metrics
SCALES_METRICS_REGISTRY: dict[str, type[MarginalMetric]] = {
    "delta_mad": DeltaMADMetric,
    "delta_mean": DeltaMeanMetric,
    "delta_median": DeltaMedianMetric,
    "delta_iqr": DeltaIQRMetric,
    "cohens_d": CohensD,
}

# Default metrics to use if none specified
DEFAULT_SCALES_METRICS = [
    "delta_mad",
    "delta_mean",
    "delta_median",
    "delta_iqr",
    "cohens_d",
]


class ScalesAggregator(BaseColumnAggregator[ScalesResult]):
    """
    Aggregator for scale distribution metrics.

    Computes multiple scale metrics across all columns, then aggregates using
    Stochastic Dominance (FSD+SSD) to produce a single quality score.

    Integrates with the reporting infrastructure via to_metric_result().

    Usage:
        aggregator = ScalesAggregator(metrics=["delta_mad", "delta_mean", "cohens_d"])
        aggregator.fit(real_df, synth_df)
        result = aggregator.compute()
        print(f"Scales Score: {result.score:.4f}")

        # For reporting integration
        metric_result = result.to_metric_result()
    """

    @classmethod
    def _get_metrics_registry(cls) -> dict[str, type[MarginalMetric]]:
        """Get the registry of available scale metrics."""
        return SCALES_METRICS_REGISTRY

    @classmethod
    def _get_default_metrics(cls) -> list[str]:
        """Get the default list of scale metrics."""
        return DEFAULT_SCALES_METRICS.copy()

    def _create_result(
        self,
        score: float,
        column_scores: dict[str, float],
        metric_details: dict[str, dict[str, ColumnMetricResult]],
        n_columns: int,
        n_metrics: int,
    ) -> ScalesResult:
        """Create a ScalesResult instance."""
        return ScalesResult(
            score=score,
            column_scores=column_scores,
            metric_details=metric_details,
            metrics_used=self.metrics_to_use,
            n_columns=n_columns,
            n_metrics=n_metrics,
        )
