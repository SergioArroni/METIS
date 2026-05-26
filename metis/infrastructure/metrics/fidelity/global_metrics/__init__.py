"""
Global fidelity metrics package.

Measures overall dataset-level similarity between real and synthetic data:
- Outliers/Coverage: Structural coverage (ranges, outliers, density)
- Correlation Matrix: Correlation structure preservation
- MMD: Maximum Mean Discrepancy (kernel-based)
- Energy Distance: Multivariate statistical distance

Usage:
    from metis.infrastructure.metrics.fidelity.global_metrics import GlobalAggregator

    aggregator = GlobalAggregator()
    result = aggregator.compute(real_df, synth_df)
    print(f"Global Score: {result.score:.4f}")
"""

from .correlation_matrix import CorrelationMatrixMetric
from .energy_distance import EnergyDistanceMetric
from .global_aggregator import DEFAULT_GLOBAL_METRICS, GLOBAL_METRICS_REGISTRY, GlobalAggregator
from .global_results import GlobalFidelityResult, GlobalMetricResult
from .mmd import MMDMetric
from .outliers_coverage import OutliersCoverageMetric

__all__ = [
    # Aggregator
    "GlobalAggregator",
    "GLOBAL_METRICS_REGISTRY",
    "DEFAULT_GLOBAL_METRICS",
    # Result structures
    "GlobalFidelityResult",
    "GlobalMetricResult",
    # Individual metrics
    "OutliersCoverageMetric",
    "CorrelationMatrixMetric",
    "MMDMetric",
    "EnergyDistanceMetric",
]
