"""
Coverage Aggregator - Orchestrates coverage metrics and computes aggregated scores.

Uses Stochastic Dominance (FSD+SSD) for principled aggregation.
Integrates with the reporting infrastructure for output generation.
"""

from metis.shared.aggregation import BaseColumnAggregator
from metis.shared.results import ColumnMetricResult

from ...fidelity_base import UniversalColumnMetric as MarginalMetric
from .coverage_result import CoverageResult
from .entropy_delta import ShannonEntropyDeltaMetric
from .gini_delta import GiniDeltaMetric
from .js import JSDivergenceMetric
from .kl import KLDivergenceMetric
from .psi import PSIMetric
from .tvd import TVDMetric

# Registry of available coverage metrics
COVERAGE_METRICS_REGISTRY: dict[str, type[MarginalMetric]] = {
    "tvd": TVDMetric,
    "kl": KLDivergenceMetric,
    "js": JSDivergenceMetric,
    "entropy_delta": ShannonEntropyDeltaMetric,
    "psi": PSIMetric,
    "gini_delta": GiniDeltaMetric,
}

# Default metrics to use if none specified
DEFAULT_COVERAGE_METRICS = ["tvd", "js", "entropy_delta", "psi"]


class CoverageAggregator(BaseColumnAggregator[CoverageResult]):
    """
    Aggregator for coverage/categorical distribution metrics.

    Computes multiple coverage metrics across all columns, then aggregates using
    Stochastic Dominance (FSD+SSD) to produce a single quality score.

    Integrates with the reporting infrastructure via to_metric_result().

    Usage:
        aggregator = CoverageAggregator(metrics=["tvd", "js", "psi"])
        aggregator.fit(real_df, synth_df)
        result = aggregator.compute()
        print(f"Coverage Score: {result.score:.4f}")

        # For reporting integration
        metric_result = result.to_metric_result()
    """

    @classmethod
    def _get_metrics_registry(cls) -> dict[str, type[MarginalMetric]]:
        """Get the registry of available coverage metrics."""
        return COVERAGE_METRICS_REGISTRY

    @classmethod
    def _get_default_metrics(cls) -> list[str]:
        """Get the default list of coverage metrics."""
        return DEFAULT_COVERAGE_METRICS.copy()

    def _create_result(
        self,
        score: float,
        column_scores: dict[str, float],
        metric_details: dict[str, dict[str, ColumnMetricResult]],
        n_columns: int,
        n_metrics: int,
    ) -> CoverageResult:
        """Create a CoverageResult instance."""
        return CoverageResult(
            score=score,
            column_scores=column_scores,
            metric_details=metric_details,
            metrics_used=self.metrics_to_use,
            n_columns=n_columns,
            n_metrics=n_metrics,
        )
