"""
Aggregation utilities for metric score consolidation.

This package provides tools for collapsing multi-dimensional metric matrices
into scalar quality scores using principled statistical methods:

- Stochastic Dominance: FSD (per-variable) and SSD (global) based aggregation
- CVaR (Conditional Value at Risk): Risk-aware averaging

Usage:
    from metis.infrastructure.metrics.aggregation import (
        aggregate_metrics,
        fsd_scores,
        ssd_score,
        normalize_metrics
    )

    # Aggregate matrix A (n variables × m metrics) to global score Q
    mu, Q = aggregate_metrics(A)
"""

from .stochastic_dominance import (
    aggregate_metrics,
    fsd_score_for_row,
    fsd_scores,
    normalize_metrics,
    ssd_score,
)

__all__ = [
    "normalize_metrics",
    "fsd_score_for_row",
    "fsd_scores",
    "ssd_score",
    "aggregate_metrics",
]
