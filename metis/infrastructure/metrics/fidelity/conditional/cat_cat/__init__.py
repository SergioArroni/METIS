"""
Categorical to Categorical (Cat↔Cat) conditional metrics package.

Computes association metrics between pairs of categorical columns:
- Cramér's V (symmetric association measure)
- Theil's U (asymmetric uncertainty coefficient)
- Chi-squared statistic (normalized)
"""

from .chi2_stat import Chi2StatMetric
from .cramers_v import CramersVMetric
from .theils_u import TheilsUMetric

__all__ = [
    "CramersVMetric",
    "TheilsUMetric",
    "Chi2StatMetric",
]
