"""
Tails metrics package - Distribution tail comparison metrics.

Available metrics:
- KS: Kolmogorov-Smirnov test
- Wasserstein: Earth Mover's distance
- Hellinger: Hellinger distance
- Anderson-Darling: AD k-sample test
- Delta Exceedance: Exceedance probability difference
- KDE-ISE: KDE Integrated Squared Error

Usage:
    from metis.infrastructure.metrics.fidelity.marginal.tails import TailsAggregator

    aggregator = TailsAggregator(metrics=["ks", "wasserstein", "hellinger"])
    aggregator.fit(real_df, synth_df)
    result = aggregator.compute()
    print(f"Tails Score: {result.score:.4f}")
"""

from metis.shared.results import ColumnMetricResult

from ...fidelity_base import NumericColumnMetric as TailMetric
from .anderson_darling import AndersonDarlingMetric
from .delta_exceedance import DeltaExceedanceMetric
from .hellinger import HellingerMetric
from .kde_ise import KDEISEMetric
from .ks import KSMetric
from .tails_aggregator import TAIL_METRICS_REGISTRY, TailsAggregator
from .tails_result import TailsResult
from .wasserstein import WassersteinMetric

__all__ = [
    # Aggregator
    "TailsAggregator",
    "TailsResult",
    "TAIL_METRICS_REGISTRY",
    # Base
    "TailMetric",
    "ColumnMetricResult",
    # Individual metrics
    "KSMetric",
    "WassersteinMetric",
    "HellingerMetric",
    "AndersonDarlingMetric",
    "DeltaExceedanceMetric",
    "KDEISEMetric",
]
