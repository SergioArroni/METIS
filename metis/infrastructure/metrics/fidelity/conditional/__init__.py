"""
Conditional fidelity metrics - bivariate relationship comparison.

Subcategories:
- Num↔Num: Correlation metrics (Pearson, Spearman, dCor, MI)
- Num↔Cat: Point-biserial, eta-squared, Kruskal-Wallis epsilon-squared
- Cat↔Cat: Cramér's V, Theil's U, Chi-squared statistic

Each subcategory has its own subpackage with individual metric implementations;
the pipeline aggregates them with ``metis.application.aggregator.Aggregator``.
"""

from .cat_cat import Chi2StatMetric, CramersVMetric, TheilsUMetric
from .num_cat import EtaSquaredMetric, KruskalEpsilonMetric, PointBiserialMetric
from .num_num import (
    DistanceCorrelationMetric,
    MutualInformationMetric,
    PearsonCorrelationMetric,
    SpearmanCorrelationMetric,
)

__all__ = [
    # Individual Num↔Num metrics
    "PearsonCorrelationMetric",
    "SpearmanCorrelationMetric",
    "DistanceCorrelationMetric",
    "MutualInformationMetric",
    # Individual Num↔Cat metrics
    "PointBiserialMetric",
    "EtaSquaredMetric",
    "KruskalEpsilonMetric",
    # Individual Cat↔Cat metrics
    "CramersVMetric",
    "TheilsUMetric",
    "Chi2StatMetric",
]
