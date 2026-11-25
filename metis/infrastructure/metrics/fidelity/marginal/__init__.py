"""
Marginal fidelity metrics.

Divided into three subcategories:
- Tails: Distribution shape and tails (KS, Wasserstein, Hellinger, AD, KDE-ISE, exceedance,
  outliers coverage)
- Scales: Where the column sits and how spread it is (Mean, Median, IQR, MAD, Cohen's d)
- Coverage: Distribution coverage metrics (TVD, KL, JS, PSI, Entropy, Gini)

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

__all__ = [
    # Base classes
    "MarginalMetric",
    "CategoricalMarginalMetric",
    "UniversalMarginalMetric",
    # Shared
    "ColumnMetricResult",
    "align_distributions",
    "get_distribution",
]
