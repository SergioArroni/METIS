"""
Tails Aggregator - Orchestrates tail metrics and computes aggregated scores.

Uses Stochastic Dominance (FSD+SSD) for principled aggregation:
- FSD (First-order Stochastic Dominance): Area under survival curve for per-column scores
- SSD (Second-order Stochastic Dominance): Double CDF integration for global score
"""

from metis.shared.aggregation import BaseColumnAggregator
from metis.shared.results import ColumnMetricResult

from ...fidelity_base import NumericColumnMetric as TailMetric
from .anderson_darling import AndersonDarlingMetric
from .delta_exceedance import DeltaExceedanceMetric
from .hellinger import HellingerMetric
from .kde_ise import KDEISEMetric
from .ks import KSMetric
from .tails_result import TailsResult
from .wasserstein import WassersteinMetric

# Registry of available tail metrics
TAIL_METRICS_REGISTRY: dict[str, type[TailMetric]] = {
    "ks": KSMetric,
    "wasserstein": WassersteinMetric,
    "hellinger": HellingerMetric,
    "anderson_darling": AndersonDarlingMetric,
    "delta_exceedance": DeltaExceedanceMetric,
    "kde_ise": KDEISEMetric,
}

# Default metrics to use if none specified
DEFAULT_TAIL_METRICS = ["ks", "wasserstein", "hellinger", "anderson_darling"]


class TailsAggregator(BaseColumnAggregator[TailsResult]):
    """
    Aggregator for tail distribution metrics.

    Computes multiple tail metrics across all columns, then aggregates using
    Stochastic Dominance (FSD+SSD) to produce a single quality score.

    Usage:
        aggregator = TailsAggregator(metrics=["ks", "wasserstein", "hellinger"])
        aggregator.fit(real_df, synth_df)
        result = aggregator.compute()
        print(f"Tails Score: {result.score:.4f}")
    """

    @classmethod
    def _get_metrics_registry(cls) -> dict[str, type[TailMetric]]:
        """Get the registry of available tail metrics."""
        return TAIL_METRICS_REGISTRY

    @classmethod
    def _get_default_metrics(cls) -> list[str]:
        """Get the default list of tail metrics."""
        return DEFAULT_TAIL_METRICS.copy()

    def _create_result(
        self,
        score: float,
        column_scores: dict[str, float],
        metric_details: dict[str, dict[str, ColumnMetricResult]],
        n_columns: int,
        n_metrics: int,
    ) -> TailsResult:
        """Create a TailsResult instance."""
        return TailsResult(
            score=score,
            column_scores=column_scores,
            metric_details=metric_details,
            metrics_used=self.metrics_to_use,
            n_columns=n_columns,
            n_metrics=n_metrics,
        )
