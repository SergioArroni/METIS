"""
Fidelity metrics for measuring synthetic data quality.

Structure:
- Global: Structural metrics (correlation matrix, MMD, energy distance)
- Marginal: Per-column distribution (Tails, Scales, Coverage)
- Conditional: Bivariate relationships (Num↔Num, Num↔Cat, Cat↔Cat)
"""

from .conditional import ConditionalAggregator, ConditionalResult
from .fidelity_aggregator import FidelityAggregator, FidelityResult
from .fidelity_base import (
    BaseFidelityMetric,
    CatCatPairMetric,
    CategoricalColumnMetric,
    ColumnFidelityMetric,
    NumCatPairMetric,
    NumericColumnMetric,
    NumNumPairMetric,
    PairFidelityMetric,
    UniversalColumnMetric,
)
from .marginal import (
    CoverageAggregator,
    CoverageResult,
    MarginalAggregator,
    MarginalResult,
    ScalesAggregator,
    ScalesResult,
    TailsAggregator,
    TailsResult,
)

__all__ = [
    # Aggregators
    "FidelityAggregator",
    "FidelityResult",
    "MarginalAggregator",
    "MarginalResult",
    "TailsAggregator",
    "TailsResult",
    "ScalesAggregator",
    "ScalesResult",
    "CoverageAggregator",
    "CoverageResult",
    "ConditionalAggregator",
    "ConditionalResult",
    # Base classes
    "BaseFidelityMetric",
    "ColumnFidelityMetric",
    "NumericColumnMetric",
    "CategoricalColumnMetric",
    "UniversalColumnMetric",
    "PairFidelityMetric",
    "NumNumPairMetric",
    "NumCatPairMetric",
    "CatCatPairMetric",
]
