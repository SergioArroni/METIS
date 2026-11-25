"""
Numeric to Categorical (Num↔Cat) conditional metrics package.

Computes association metrics between numeric and categorical columns:
- Point-biserial correlation (for binary categories)
- Eta-squared (effect size from ANOVA)
- Kruskal-Wallis epsilon-squared (non-parametric)
"""

from .eta_squared import EtaSquaredMetric
from .kruskal_epsilon import KruskalEpsilonMetric
from .point_biserial import PointBiserialMetric

__all__ = [
    "PointBiserialMetric",
    "EtaSquaredMetric",
    "KruskalEpsilonMetric",
]
