"""
Scales metrics package - Distributional scale comparison metrics.

Available metrics:
- DeltaMAD: Delta Median Absolute Deviation
- DeltaMean: Delta Mean
- DeltaMedian: Delta Median
- DeltaIQR: Delta Interquartile Range
- Cohen's d: Effect size measure

Usage:
    from metis.infrastructure.metrics.fidelity.marginal.scales import ScalesAggregator

    aggregator = ScalesAggregator(metrics=["delta_mad", "delta_mean", "cohens_d"])
    aggregator.fit(real_df, synth_df)
    result = aggregator.compute()
    print(f"Scales Score: {result.score:.4f}")

    # For reporting integration
    metric_result = result.to_metric_result()
"""

from .cohens_d import CohensD
from .delta_iqr import DeltaIQRMetric
from .delta_mad import DeltaMADMetric
from .delta_mean import DeltaMeanMetric
from .delta_median import DeltaMedianMetric
from .scales_aggregator import DEFAULT_SCALES_METRICS, SCALES_METRICS_REGISTRY, ScalesAggregator
from .scales_result import ScalesResult

__all__ = [
    # Aggregator
    "ScalesAggregator",
    "ScalesResult",
    "SCALES_METRICS_REGISTRY",
    "DEFAULT_SCALES_METRICS",
    # Individual metrics
    "DeltaMADMetric",
    "DeltaMeanMetric",
    "DeltaMedianMetric",
    "DeltaIQRMetric",
    "CohensD",
]
