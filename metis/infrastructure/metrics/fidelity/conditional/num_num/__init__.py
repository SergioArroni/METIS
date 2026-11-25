"""
Numeric to Numeric (Num↔Num) conditional metrics package.

Computes correlation metrics between pairs of numeric columns:
- Pearson correlation (linear)
- Spearman correlation (rank-based)
- Distance correlation (dCor) - captures non-linear dependencies
- Mutual information (MI) - information-theoretic measure
- Correlation matrix - the whole matrix of numeric correlations
"""

from .correlation_matrix import CorrelationMatrixMetric
from .dcor import DistanceCorrelationMetric
from .mi import MutualInformationMetric
from .pearson import PearsonCorrelationMetric
from .spearman import SpearmanCorrelationMetric

__all__ = [
    "PearsonCorrelationMetric",
    "SpearmanCorrelationMetric",
    "DistanceCorrelationMetric",
    "MutualInformationMetric",
    "CorrelationMatrixMetric",
]
