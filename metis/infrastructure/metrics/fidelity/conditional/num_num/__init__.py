"""
Numeric to Numeric (Num↔Num) conditional metrics package.

Computes correlation metrics between pairs of numeric columns:
- Pearson correlation (linear)
- Spearman correlation (rank-based)
- Distance correlation (dCor) - captures non-linear dependencies
- Mutual information (MI) - information-theoretic measure
"""

from .dcor import DistanceCorrelationMetric
from .mi import MutualInformationMetric
from .num_num_aggregator import NumNumMetrics
from .pearson import PearsonCorrelationMetric
from .spearman import SpearmanCorrelationMetric

__all__ = [
    "NumNumMetrics",
    "PearsonCorrelationMetric",
    "SpearmanCorrelationMetric",
    "DistanceCorrelationMetric",
    "MutualInformationMetric",
]
