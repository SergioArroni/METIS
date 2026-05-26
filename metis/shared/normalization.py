"""
Normalization utilities for metrics.

Provides specific normalization functions for each metric type.
All normalizations return values in [0, 1] where 1 = best quality.
"""

import math
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np

# =============================================================================
# CORE UTILITIES
# =============================================================================


def clamp(x: float) -> float:
    """Clamp value to [0, 1] range."""
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))


# =============================================================================
# NORMALIZATION TYPES
# =============================================================================


class NormalizationType(Enum):
    """Types of normalization strategies."""

    # 1. Lower = better, bounded [0, 1]
    BOUNDED_DISTANCE = "bounded_distance"

    # 2. Lower = better, unbounded (uses exponential decay)
    UNBOUNDED_DISTANCE = "unbounded_distance"

    # 3. Higher = better, already in [0, 1]
    SIMILARITY = "similarity"

    # 4. Delta/difference metrics (uses inverse)
    DELTA = "delta"

    # 5. Mutual Information (special scaling)
    MUTUAL_INFORMATION = "mutual_information"

    # 6. DCR - special case (higher = better, unbounded)
    DCR = "dcr"

    # 7. ML loss (lower = better)
    ML_LOSS = "ml_loss"


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================


def normalize_bounded_distance(value: float, _params: dict[str, Any] | None = None) -> float:
    """
    Normalize bounded distance metrics where lower = better.

    For metrics like KS, Hellinger, TVD, JS that are naturally in [0, 1].

    Args:
        value: Raw metric value in [0, 1] where 0 = best
        _params: Optional parameters (not used)

    Returns:
        Normalized value in [0, 1] where 1 = best
    """
    return clamp(1.0 - value)


def normalize_unbounded_distance(value: float, params: dict[str, Any] | None = None) -> float:
    """
    Normalize unbounded distance metrics where lower = better.

    Uses exponential decay: exp(-value / scale).
    For metrics like KL, Wasserstein, MMD, Energy Distance.

    Args:
        value: Raw metric value >= 0 where 0 = best
        params: Optional dict with 'scale' parameter (default 1.0)

    Returns:
        Normalized value in [0, 1] where 1 = best
    """
    params = params or {}
    scale = params.get("scale", 1.0)
    if scale <= 0:
        scale = 1.0
    return clamp(math.exp(-value / scale))


def normalize_similarity(value: float, _params: dict[str, Any] | None = None) -> float:
    """
    Normalize bounded score metrics where higher = better.

    For metrics like Pearson, Spearman, Cramer's V, F1, accuracy, or AUC
    that are already in [0, 1].

    Args:
        value: Raw metric value in [0, 1] where 1 = best
        _params: Optional parameters (not used)

    Returns:
        Normalized value in [0, 1] where 1 = best
    """
    return clamp(value)


def normalize_delta(value: float, _params: dict[str, Any] | None = None) -> float:
    """
    Normalize delta/difference metrics.

    Uses inverse: 1 / (1 + |value|).
    For metrics like delta_iqr, delta_mad, entropy_delta, gini_delta.

    Args:
        value: Raw delta value where 0 = best
        _params: Optional parameters (not used)

    Returns:
        Normalized value in [0, 1] where 1 = best
    """
    return clamp(1.0 / (1.0 + abs(value)))


def normalize_mutual_information(value: float, params: dict[str, Any] | None = None) -> float:
    """
    Normalize mutual information metrics.

    Uses: value / (value + scale).
    MI can be unbounded, this provides smooth normalization.

    Args:
        value: Raw MI value >= 0
        params: Optional dict with 'mi_scale' parameter (default 1.0)

    Returns:
        Normalized value in [0, 1] where 1 = high mutual information
    """
    params = params or {}
    scale = params.get("mi_scale", 1.0)
    if scale <= 0:
        scale = 1.0
    return clamp(value / (value + scale))


def normalize_dcr(value: float, params: dict[str, Any] | None = None) -> float:
    """
    Normalize Distance to Closest Record (DCR).

    DCR is unbounded and higher = better (more privacy).
    Uses: 1 - exp(-value / scale).

    Args:
        value: Raw DCR value >= 0 where higher = more private
        params: Optional dict with 'scale' parameter (default 10.0)

    Returns:
        Normalized value in [0, 1] where 1 = best privacy
    """
    params = params or {}
    scale = params.get("scale", 10.0)
    if scale <= 0:
        scale = 10.0
    return clamp(1.0 - math.exp(-value / scale))


def normalize_ml_loss(value: float, _params: dict[str, Any] | None = None) -> float:
    """
    Normalize ML loss metrics where lower = better.

    Uses: 1 / (1 + value).
    For metrics like MAE, RMSE, loss values.

    Args:
        value: Raw loss value >= 0 where 0 = best
        _params: Optional parameters (not used)

    Returns:
        Normalized value in [0, 1] where 1 = best
    """
    return clamp(1.0 / (1.0 + value))


# =============================================================================
# NORMALIZATION REGISTRY
# =============================================================================

# Map normalization type to function
NORMALIZATION_FUNCTIONS: dict[NormalizationType, Callable[[float, dict | None], float]] = {
    NormalizationType.BOUNDED_DISTANCE: normalize_bounded_distance,
    NormalizationType.UNBOUNDED_DISTANCE: normalize_unbounded_distance,
    NormalizationType.SIMILARITY: normalize_similarity,
    NormalizationType.DELTA: normalize_delta,
    NormalizationType.MUTUAL_INFORMATION: normalize_mutual_information,
    NormalizationType.DCR: normalize_dcr,
    NormalizationType.ML_LOSS: normalize_ml_loss,
}

# Metric ID to normalization type mapping
METRIC_NORMALIZATION_MAP: dict[str, NormalizationType] = {
    # ==========================================================================
    # FIDELITY - Global
    # ==========================================================================
    "fidelity.correlation_matrix": NormalizationType.SIMILARITY,
    "fidelity.mmd": NormalizationType.UNBOUNDED_DISTANCE,
    "fidelity.energy_distance": NormalizationType.UNBOUNDED_DISTANCE,
    "fidelity.outliers_coverage": NormalizationType.SIMILARITY,
    # ==========================================================================
    # FIDELITY - Marginal - Tails
    # ==========================================================================
    "fidelity.ks": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.wasserstein": NormalizationType.UNBOUNDED_DISTANCE,
    "fidelity.anderson_darling": NormalizationType.UNBOUNDED_DISTANCE,
    "fidelity.hellinger": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.kde_ise": NormalizationType.UNBOUNDED_DISTANCE,
    "fidelity.delta_exceedance": NormalizationType.DELTA,
    # ==========================================================================
    # FIDELITY - Marginal - Scales
    # ==========================================================================
    "fidelity.delta_mean": NormalizationType.DELTA,
    "fidelity.delta_median": NormalizationType.DELTA,
    "fidelity.delta_iqr": NormalizationType.DELTA,
    "fidelity.delta_mad": NormalizationType.DELTA,
    "fidelity.cohens_d": NormalizationType.DELTA,
    # ==========================================================================
    # FIDELITY - Marginal - Coverage
    # ==========================================================================
    "fidelity.tvd": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.js": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.kl": NormalizationType.UNBOUNDED_DISTANCE,
    "fidelity.psi": NormalizationType.UNBOUNDED_DISTANCE,
    "fidelity.entropy_delta": NormalizationType.DELTA,
    "fidelity.gini_delta": NormalizationType.DELTA,
    # ==========================================================================
    # FIDELITY - Conditional - Num↔Num
    # ==========================================================================
    # These conditional metrics are aggregated from pairwise deltas |real - synth|,
    # so their normalization must treat the input as a distance, not as a raw
    # association score.
    "fidelity.pearson": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.spearman": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.dcor": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.mi": NormalizationType.UNBOUNDED_DISTANCE,
    # ==========================================================================
    # FIDELITY - Conditional - Num↔Cat
    # ==========================================================================
    "fidelity.eta_squared": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.point_biserial": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.kruskal_epsilon": NormalizationType.BOUNDED_DISTANCE,
    # ==========================================================================
    # FIDELITY - Conditional - Cat↔Cat
    # ==========================================================================
    "fidelity.cramers_v": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.theils_u": NormalizationType.BOUNDED_DISTANCE,
    "fidelity.chi2_stat": NormalizationType.UNBOUNDED_DISTANCE,
    # ==========================================================================
    # PRIVACY
    # ==========================================================================
    "privacy.dcr": NormalizationType.DCR,
    "privacy.nnaa": NormalizationType.SIMILARITY,
    "privacy.mia": NormalizationType.SIMILARITY,
    "privacy.inference_attack": NormalizationType.SIMILARITY,
    "privacy.k_anonymity": NormalizationType.SIMILARITY,
    "privacy.l_diversity": NormalizationType.SIMILARITY,
    "privacy.t_closeness": NormalizationType.SIMILARITY,
    "privacy.record_linkage": NormalizationType.SIMILARITY,
    "privacy.differential_privacy": NormalizationType.SIMILARITY,
    # ==========================================================================
    # UTILITY
    # ==========================================================================
    "utility.classification": NormalizationType.SIMILARITY,
    "utility.regression": NormalizationType.SIMILARITY,
    "utility.auto": NormalizationType.SIMILARITY,
    "utility.ml_efficiency": NormalizationType.SIMILARITY,
    "utility.classification_efficiency": NormalizationType.SIMILARITY,
    "utility.regression_efficiency": NormalizationType.SIMILARITY,
}

# Default parameters for metrics that need them
METRIC_NORMALIZATION_PARAMS: dict[str, dict[str, Any]] = {
    # Unbounded distances - scale based on typical ranges
    "fidelity.wasserstein": {"scale": 0.5},
    "fidelity.mmd": {"scale": 0.1},
    "fidelity.energy_distance": {"scale": 1.0},
    "fidelity.anderson_darling": {"scale": 5.0},
    "fidelity.kde_ise": {"scale": 0.1},
    "fidelity.kl": {"scale": 1.0},
    "fidelity.psi": {"scale": 0.25},
    "fidelity.chi2_stat": {"scale": 100.0},
    # Mutual information
    "fidelity.mi": {"mi_scale": 1.0},
    # DCR
    "privacy.dcr": {"scale": 10.0},
}


def get_normalizer(metric_id: str) -> Callable[[float, dict | None], float]:
    """
    Get the normalization function for a metric.

    Args:
        metric_id: Metric identifier (e.g., "fidelity.ks")

    Returns:
        Normalization function
    """
    # Try exact match first
    if metric_id in METRIC_NORMALIZATION_MAP:
        norm_type = METRIC_NORMALIZATION_MAP[metric_id]
        return NORMALIZATION_FUNCTIONS[norm_type]

    # Try without prefix
    short_id = metric_id.split(".")[-1] if "." in metric_id else metric_id
    for full_id, norm_type in METRIC_NORMALIZATION_MAP.items():
        if full_id.endswith(f".{short_id}"):
            return NORMALIZATION_FUNCTIONS[norm_type]

    # Default to similarity (identity with clamp)
    return normalize_similarity


def get_normalization_params(metric_id: str) -> dict[str, Any]:
    """
    Get default normalization parameters for a metric.

    Args:
        metric_id: Metric identifier

    Returns:
        Dictionary of parameters
    """
    return METRIC_NORMALIZATION_PARAMS.get(metric_id, {})


def normalize_metric_value(
    metric_id: str,
    value: float,
    params: dict[str, Any] | None = None,
) -> float:
    """
    Normalize a metric value using the appropriate strategy.

    Args:
        metric_id: Metric identifier (e.g., "fidelity.ks")
        value: Raw metric value
        params: Optional override parameters

    Returns:
        Normalized value in [0, 1] where 1 = best
    """
    normalizer = get_normalizer(metric_id)
    default_params = get_normalization_params(metric_id)

    # Merge params (provided params override defaults)
    if params:
        final_params = {**default_params, **params}
    else:
        final_params = default_params

    return normalizer(value, final_params)


# =============================================================================
# BATCH NORMALIZATION (for column-wise metrics)
# =============================================================================


def normalize_values_batch(
    raw_values: dict[str, float],
    metric_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Normalize a batch of values for a given metric.

    Args:
        raw_values: Dictionary of column/pair names to raw values
        metric_id: Metric identifier
        params: Optional override parameters

    Returns:
        Dictionary of normalized values in [0, 1]
    """
    normalizer = get_normalizer(metric_id)
    default_params = get_normalization_params(metric_id)
    final_params = {**default_params, **(params or {})}

    return {
        key: normalizer(value, final_params) if not np.isnan(value) else float("nan")
        for key, value in raw_values.items()
    }
