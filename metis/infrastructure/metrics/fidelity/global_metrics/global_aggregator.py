"""
Global Metrics Aggregator.

Aggregates results from all global metrics into a single score:
- Outliers/Coverage
- Correlation Matrix Similarity
- MMD
- Energy Distance
"""

import math

import pandas as pd

from .correlation_matrix import CorrelationMatrixMetric
from .energy_distance import EnergyDistanceMetric
from .global_results import GlobalFidelityResult, GlobalMetricResult
from .mmd import MMDMetric
from .outliers_coverage import OutliersCoverageMetric

# Registry of available global metrics
GLOBAL_METRICS_REGISTRY = {
    "outliers_coverage": OutliersCoverageMetric,
    "correlation_matrix": CorrelationMatrixMetric,
    "mmd": MMDMetric,
    "energy_distance": EnergyDistanceMetric,
}

DEFAULT_GLOBAL_METRICS = [
    "outliers_coverage",
    "correlation_matrix",
    "mmd",
    "energy_distance",
]


class GlobalAggregator:
    """
    Aggregator for global/dataset-level fidelity metrics.

    Computes how well synthetic data preserves global properties
    of the original dataset:
    - Structural coverage (outliers, ranges, density)
    - Correlation structure preservation
    - Distribution similarity (MMD, Energy Distance)

    The final score is a weighted average of all metrics.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> real = pd.DataFrame(np.random.randn(1000, 10))
        >>> synth = pd.DataFrame(np.random.randn(1000, 10))
        >>> aggregator = GlobalAggregator()
        >>> result = aggregator.compute(real, synth)
        >>> print(f"Global Score: {result.score:.4f}")
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ):
        """
        Initialize the aggregator.

        Args:
            metrics: list of metric names to use. If None, uses defaults.
            weights: Custom weights for each metric. If None, uses equal weights.
        """
        if metrics is None:
            metrics = DEFAULT_GLOBAL_METRICS.copy()

        for m in metrics:
            if m not in GLOBAL_METRICS_REGISTRY:
                available = list(GLOBAL_METRICS_REGISTRY.keys())
                raise ValueError(f"Unknown metric '{m}'. Available: {available}")

        self.metrics_to_use = metrics
        if weights is None:
            self.weights = {m: 1.0 / len(metrics) for m in metrics}
        else:
            for m, w in weights.items():
                if not math.isfinite(w) or w < 0:
                    raise ValueError(f"weight for metric '{m}' must be finite and >= 0 (got {w})")
            if sum(weights.values()) <= 0:
                raise ValueError("sum of metric weights must be > 0")
            self.weights = weights

        self._result: GlobalFidelityResult | None = None

    @classmethod
    def available_metrics(cls) -> list[str]:
        """Return list of available metric names."""
        return list(GLOBAL_METRICS_REGISTRY.keys())

    def compute(self, real_data: pd.DataFrame, synth_data: pd.DataFrame) -> GlobalFidelityResult:
        """
        Compute all global metrics and aggregate.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset

        Returns:
            GlobalFidelityResult with aggregated scores
        """
        metric_details: dict[str, GlobalMetricResult] = {}

        # Compute each metric
        for metric_name in self.metrics_to_use:
            metric_class = GLOBAL_METRICS_REGISTRY[metric_name]
            metric = metric_class()
            result = metric.compute(real_data, synth_data)
            metric_details[metric_name] = result

        # Extract scores
        outliers_score = metric_details.get(
            "outliers_coverage",
            GlobalMetricResult.invalid("outliers_coverage", "Not computed"),
        ).normalized_value

        correlation_score = metric_details.get(
            "correlation_matrix",
            GlobalMetricResult.invalid("correlation_matrix", "Not computed"),
        ).normalized_value

        mmd_score = metric_details.get(
            "mmd", GlobalMetricResult.invalid("mmd", "Not computed")
        ).normalized_value

        energy_score = metric_details.get(
            "energy_distance",
            GlobalMetricResult.invalid("energy_distance", "Not computed"),
        ).normalized_value

        # Compute weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for metric_name, result in metric_details.items():
            if result.is_valid:
                weight = self.weights.get(metric_name, 1.0 / len(self.metrics_to_use))
                weighted_sum += weight * result.normalized_value
                total_weight += weight

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        n_computed = sum(1 for r in metric_details.values() if r.is_valid)

        self._result = GlobalFidelityResult(
            score=final_score,
            outliers_coverage_score=outliers_score,
            correlation_matrix_score=correlation_score,
            mmd_score=mmd_score,
            energy_distance_score=energy_score,
            metric_details=metric_details,
            n_metrics_computed=n_computed,
        )

        return self._result

    @property
    def result(self) -> GlobalFidelityResult | None:
        """Get the last computed result."""
        return self._result

    def get_detailed_report(self) -> dict:
        """
        Get a detailed report of all metrics.

        Returns:
            Dictionary with detailed breakdown
        """
        if self._result is None:
            raise ValueError("Must call compute() first")

        report = {
            "overall_score": self._result.score,
            "metrics": {},
        }

        for name, result in self._result.metric_details.items():
            report["metrics"][name] = {
                "score": result.normalized_value,
                "raw_value": result.raw_value,
                "is_valid": result.is_valid,
                "details": result.details,
            }

        return report
