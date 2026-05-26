"""
Coverage metrics package - Categorical distribution and coverage comparison metrics.

Available metrics:
- TVD: Total Variation Distance
- KL: Kullback-Leibler Divergence
- JS: Jensen-Shannon Divergence
- Shannon Entropy Delta
- PSI: Population Stability Index
- Gini Delta: Gini coefficient comparison

Usage:
    from metis.infrastructure.metrics.fidelity.marginal.coverage import CoverageAggregator

    aggregator = CoverageAggregator(metrics=["tvd", "js", "psi"])
    aggregator.fit(real_df, synth_df)
    result = aggregator.compute()
    print(f"Coverage Score: {result.score:.4f}")

    # For reporting integration
    metric_result = result.to_metric_result()
"""

from .coverage_aggregator import (
    COVERAGE_METRICS_REGISTRY,
    DEFAULT_COVERAGE_METRICS,
    CoverageAggregator,
)
from .coverage_result import CoverageResult
from .entropy_delta import ShannonEntropyDeltaMetric
from .gini_delta import GiniDeltaMetric
from .js import JSDivergenceMetric
from .kl import KLDivergenceMetric
from .psi import PSIMetric
from .tvd import TVDMetric

__all__ = [
    # Aggregator
    "CoverageAggregator",
    "CoverageResult",
    "COVERAGE_METRICS_REGISTRY",
    "DEFAULT_COVERAGE_METRICS",
    # Individual metrics
    "TVDMetric",
    "KLDivergenceMetric",
    "JSDivergenceMetric",
    "ShannonEntropyDeltaMetric",
    "PSIMetric",
    "GiniDeltaMetric",
]
