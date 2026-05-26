"""
Centralized registry of aggregation functions for METIS.

Provides a single source of truth for aggregation functions used across:
- Calibration (aggregator_tuner.py)
- Main aggregation (application/aggregator.py)
- Custom metric implementations

This eliminates code duplication and ensures consistency across the codebase.

Performance Notes:
    - FSD (First-Order Stochastic Dominance): O(n) - arithmetic mean
    - SSD (Second-Order Stochastic Dominance): O(n) - exponential utility
    - Geometric/Harmonic mean: O(n)
    - Trimmed mean: O(n log n) - requires sorting
    - Min/Max/Percentiles: O(n) for single pass, O(n log n) with sorting
"""

from collections.abc import Callable

import numpy as np
from scipy import stats

from metis.infrastructure.metrics.aggregation.stochastic_dominance import (
    fsd_score_for_row,
    ssd_score,
)


def _strip_nan(x):
    """Remove NaN/Inf values from input before aggregation."""
    arr = np.asarray(x, dtype=np.float64)
    return arr[np.isfinite(arr)]


# Core aggregation functions registry
# All functions are NaN-safe: non-finite values are silently dropped
# before aggregation.  If all values are non-finite the function
# returns NaN so the caller can handle it explicitly.
AGGREGATION_FUNCTIONS: dict[str, Callable[[list], float]] = {
    # Basic statistical aggregators
    "mean": lambda x: float(np.mean(v)) if len(v := _strip_nan(x)) else float("nan"),
    "median": lambda x: float(np.median(v)) if len(v := _strip_nan(x)) else float("nan"),
    "min": lambda x: float(np.min(v)) if len(v := _strip_nan(x)) else float("nan"),
    "max": lambda x: float(np.max(v)) if len(v := _strip_nan(x)) else float("nan"),
    # Percentile-based aggregators
    "percentile_25": lambda x: (
        float(np.percentile(v, 25)) if len(v := _strip_nan(x)) else float("nan")
    ),
    "percentile_75": lambda x: (
        float(np.percentile(v, 75)) if len(v := _strip_nan(x)) else float("nan")
    ),
    # Robust aggregators (trimmed means)
    "trimmed_mean_10": lambda x: (
        float(stats.trim_mean(v, 0.1)) if len(v := _strip_nan(x)) else float("nan")
    ),
    "trimmed_mean_20": lambda x: (
        float(stats.trim_mean(v, 0.2)) if len(v := _strip_nan(x)) else float("nan")
    ),
    # Mean variants (geometric/harmonic)
    "geometric_mean": lambda x: (
        (float(stats.gmean(v)) if np.all(v > 0) else 0.0)
        if len(v := _strip_nan(x))
        else float("nan")
    ),
    "harmonic_mean": lambda x: (
        (float(stats.hmean(v)) if np.all(v > 0) else 0.0)
        if len(v := _strip_nan(x))
        else float("nan")
    ),
    # Stochastic dominance aggregators
    # FSD: First-order (mean-based, less risk-averse)
    "fsd": lambda x: fsd_score_for_row(v) if len(v := _strip_nan(x)) else float("nan"),
    # SSD: Second-order (exponential utility, risk-averse)
    # Default risk_aversion=5.0 (moderate)
    "ssd": lambda x: ssd_score(v, risk_aversion=5.0) if len(v := _strip_nan(x)) else float("nan"),
}


def get_aggregation_function(name: str, **kwargs) -> Callable[[list], float]:
    """
    Get aggregation function by name with optional parameters.

    Args:
        name: Name of aggregation function
        **kwargs: Optional parameters (e.g., risk_aversion for SSD)

    Returns:
        Aggregation function

    Raises:
        KeyError: If aggregation function not found

    Examples:
        >>> agg_fn = get_aggregation_function("mean")
        >>> agg_fn([0.8, 0.9, 0.7])
        0.8

        >>> agg_fn = get_aggregation_function("ssd", risk_aversion=7.0)
        >>> agg_fn([0.8, 0.9, 0.7])
        0.78  # More conservative than mean
    """
    if name not in AGGREGATION_FUNCTIONS:
        available = ", ".join(sorted(AGGREGATION_FUNCTIONS.keys()))
        raise KeyError(f"Unknown aggregation function: '{name}'. Available: {available}")

    base_fn = AGGREGATION_FUNCTIONS[name]

    # Handle parameterized functions
    if name == "ssd" and "risk_aversion" in kwargs:
        risk_aversion = kwargs["risk_aversion"]
        return lambda x: (
            ssd_score(v, risk_aversion=risk_aversion) if len(v := _strip_nan(x)) else float("nan")
        )

    return base_fn


def list_aggregation_functions() -> list:
    """
    list all available aggregation functions.

    Returns:
        Sorted list of function names
    """
    return sorted(AGGREGATION_FUNCTIONS.keys())


# Aggregation function categories for documentation/UI
AGGREGATION_CATEGORIES = {
    "basic": ["mean", "median", "min", "max"],
    "percentile": ["percentile_25", "percentile_75"],
    "robust": ["trimmed_mean_10", "trimmed_mean_20"],
    "mean_variants": ["geometric_mean", "harmonic_mean"],
    "stochastic_dominance": ["fsd", "ssd"],
}


def get_aggregation_category(name: str) -> str:
    """
    Get category for an aggregation function.

    Args:
        name: Function name

    Returns:
        Category name or "other" if not categorized
    """
    for category, functions in AGGREGATION_CATEGORIES.items():
        if name in functions:
            return category
    return "other"
