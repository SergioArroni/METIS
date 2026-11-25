"""
Fidelity metrics for measuring synthetic data quality.

Structure:
- Global: Structural metrics (correlation matrix, MMD, energy distance, outliers)
- Marginal: Per-column distribution (Tails, Scales, Coverage)
- Conditional: Bivariate relationships (Num↔Num, Num↔Cat, Cat↔Cat)

Metrics are aggregated by ``metis.application.aggregator.Aggregator``
(metrics -> subcategory -> category -> family). The per-package aggregators
(``FidelityAggregator``, ``MarginalAggregator``...) were removed: nothing used
them, they did not receive the execution ``context`` and they reimplemented
the metrics with formulas that had drifted from the registered ones.
"""

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

__all__ = [
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
