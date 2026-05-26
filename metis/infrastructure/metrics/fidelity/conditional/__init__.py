"""
Conditional fidelity metrics - bivariate relationship comparison.

Subcategories:
- Num↔Num: Correlation metrics (Pearson, Spearman, dCor, MI)
- Num↔Cat: Point-biserial, ANOVA F, eta-squared, Kruskal-Wallis
- Cat↔Cat: Cramér's V, Theil's U, Chi-squared independence

Each subcategory has its own subpackage with individual metric implementations.
"""

from .cat_cat import CatCatMetrics, Chi2StatMetric, CramersVMetric, TheilsUMetric
from .conditional_aggregator import ConditionalAggregator
from .conditional_result import ConditionalResult
from .num_cat import EtaSquaredMetric, KruskalEpsilonMetric, NumCatMetrics, PointBiserialMetric

# Import individual metrics from subcategories
from .num_num import (
    DistanceCorrelationMetric,
    MutualInformationMetric,
    NumNumMetrics,
    PearsonCorrelationMetric,
    SpearmanCorrelationMetric,
)
from .pair_results import PairMetricResult

__all__ = [
    # Aggregated metric classes
    "NumNumMetrics",
    "NumCatMetrics",
    "CatCatMetrics",
    # Aggregator
    "ConditionalAggregator",
    # Result structures
    "ConditionalResult",
    "PairMetricResult",
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
