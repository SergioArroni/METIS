"""
Marginal fidelity metrics.

Divided into three subcategories:
- Tails: Distribution tail metrics (KS, Wasserstein, Hellinger, AD, etc.)
- Scales: Central tendency and dispersion (Mean, Median, MAD, IQR, Cohen's d)
- Coverage: Categorical distribution metrics (TVD, KL, JS, PSI, Entropy)

Shared utilities:
- ColumnMetricResult: Common result structure for all marginal metrics
- Distribution utilities: align_distributions, get_distribution
"""

from metis.shared import ColumnMetricResult, align_distributions, get_distribution

from ..fidelity_base import (
    CategoricalColumnMetric as CategoricalMarginalMetric,
    NumericColumnMetric as MarginalMetric,
    UniversalColumnMetric as UniversalMarginalMetric,
)
from .coverage import CoverageAggregator, CoverageResult
from .marginal_aggregator import MarginalAggregator
from .marginal_result import MarginalResult
from .scales import ScalesAggregator, ScalesResult
from .tails import TailsAggregator, TailsResult

__all__ = [
    # Aggregators
    "TailsAggregator",
    "TailsResult",
    "ScalesAggregator",
    "ScalesResult",
    "CoverageAggregator",
    "CoverageResult",
    "MarginalAggregator",
    "MarginalResult",
    # Base classes
    "MarginalMetric",
    "CategoricalMarginalMetric",
    "UniversalMarginalMetric",
    # Shared
    "ColumnMetricResult",
    "align_distributions",
    "get_distribution",
]
