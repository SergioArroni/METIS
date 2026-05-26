"""
Stochastic Dominance-based Score Aggregation.

This module implements a hierarchical aggregation pipeline using stochastic dominance
concepts for multi-level metric aggregation:

Aggregation Hierarchy:
    Level 1: Individual metrics → Subcategory score (FSD)
    Level 2: Subcategories → Category score (FSD)
    Level 3: Categories → Domain score (SSD)
    Level 4: Domains → Final validation score (SSD)

FSD (First-Order Stochastic Dominance):
    - Used for lower levels (1-2) where we want to capture average behavior
    - Area under survival curve S(t) = P(X ≥ t)
    - Less penalizing of variance

SSD (Second-Order Stochastic Dominance):
    - Used for higher levels (3-4) where we want risk-averse aggregation
    - Double integration of CDF penalizes poor performance more heavily
    - Ensures worst-case scenarios significantly impact the final score

References:
    - Levy, H. (2016). Stochastic Dominance: Investment Decision Making under Uncertainty
    - Quiroga, R. & Egozcue, M. (2008). On Consistent Tests of Stochastic Dominance
"""

from enum import Enum

import numpy as np


class AggregationLevel(Enum):
    """
    Defines the aggregation level and corresponding method.

    FSD is used for lower levels (less penalizing), SSD for higher levels (risk-averse).
    """

    LEVEL_1_METRICS = 1  # Individual metrics → Subcategory (FSD)
    LEVEL_2_SUBCATEGORY = 2  # Subcategories → Category (FSD)
    LEVEL_3_CATEGORY = 3  # Categories → Domain (SSD)
    LEVEL_4_FINAL = 4  # Domains → Final score (SSD)


def normalize_metrics(
    A: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    lower_is_better: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalize a metrics matrix to [0, 1] using robust percentile scaling.

    For each column (metric), applies:
    1. Clip values to [p5, p95] to handle outliers
    2. Linear rescaling to [0, 1]
    3. If metric is "lower is better", apply 1 - normalized

    Args:
        A: Matrix of shape (n, m) where n = variables, m = metrics
        lower_percentile: Lower percentile for clipping (default 5)
        upper_percentile: Upper percentile for clipping (default 95)
        lower_is_better: Boolean array of shape (m,) indicating which metrics
                        should be inverted. If None, assumes all metrics are
                        "higher is better".

    Returns:
        Normalized matrix with all values in [0, 1], where 1 = best

    Example:
        >>> A = np.array([[10, 0.5], [20, 0.3], [15, 0.8]])
        >>> # First column: higher is better. Second column: lower is better
        >>> lower_is_better = np.array([False, True])
        >>> A_norm = normalize_metrics(A, lower_is_better=lower_is_better)
    """
    A = np.asarray(A, dtype=np.float64)
    _, m = A.shape

    if lower_is_better is None:
        lower_is_better = np.zeros(m, dtype=bool)

    A_normalized = np.zeros_like(A)

    for j in range(m):
        col = A[:, j]

        # Compute robust bounds
        p_low = np.percentile(col, lower_percentile)
        p_high = np.percentile(col, upper_percentile)

        # Avoid division by zero
        if p_high - p_low < 1e-10:
            # Constant column: pass the value through (clipped to [0, 1])
            # instead of forcing 0.5. The previous neutral default flattened
            # truly perfect (1.0) and truly failed (0.0) constant columns.
            const_val = float(np.clip(np.nanmean(col), 0.0, 1.0))
            if lower_is_better[j]:
                const_val = 1.0 - const_val
            A_normalized[:, j] = const_val
        else:
            # Clip and normalize
            clipped = np.clip(col, p_low, p_high)
            normalized = (clipped - p_low) / (p_high - p_low)

            # Invert if lower is better
            if lower_is_better[j]:
                normalized = 1.0 - normalized

            A_normalized[:, j] = normalized

    return np.clip(A_normalized, 0.0, 1.0)


def fsd_score_for_row(row: np.ndarray) -> float:
    """
    Compute First-Order Stochastic Dominance (FSD) score for a single row.

    FSD criterion is equivalent to comparing expected values. A distribution F
    first-order stochastically dominates G if and only if:
        E_F[X] ≥ E_G[X]

    Therefore, the FSD score is simply the arithmetic mean:
        μ_FSD = E[X] = (1/m) Σᵢ xᵢ

    This is mathematically equivalent to the area under the survival curve
    S(t) = P(X ≥ t) for discrete distributions:
        ∫₀¹ S(t) dt = E[X]

    Interpretation:
        - μ = 1.0: All metrics are perfect (equal to 1)
        - μ = 0.5: Average metric value is 0.5
        - μ = 0.0: All metrics are at their worst (equal to 0)

    Args:
        row: Array of metric values for a single variable, already in [0, 1]

    Returns:
        FSD score μ ∈ [0, 1] (arithmetic mean)

    Mathematical Background:
        For a random variable X with values in [0,1], the FSD criterion states
        that distribution F dominates G if F(t) ≤ G(t) for all t ∈ [0,1].
        This is equivalent to E_F[X] ≥ E_G[X].

    References:
        - Levy, H. (2016). Stochastic Dominance: Investment Decision Making
        - Quirk, J. P., & Saposnik, R. (1962). Admissibility and Measurable Utility Functions
    """
    row = np.asarray(row, dtype=np.float64)
    m = len(row)

    if m == 0:
        return 0.0

    # FSD score is simply the arithmetic mean
    mu = np.mean(row)

    return float(np.clip(mu, 0.0, 1.0))


def fsd_scores(A: np.ndarray) -> np.ndarray:
    """
    Compute FSD scores for all rows (variables) in the matrix.

    FSD scores are computed as the arithmetic mean of each row, which is
    mathematically equivalent to the area under the survival curve.
    This vectorized implementation is ~10x faster than row-by-row computation.

    Args:
        A: Normalized matrix of shape (n, m), values in [0, 1]

    Returns:
        Array μ of shape (n,) with FSD scores for each variable

    Example:
        >>> A = np.array([[0.9, 0.8, 0.85], [0.3, 0.4, 0.35]])
        >>> mu = fsd_scores(A)
        >>> # mu[0] ≈ 0.85 (high scores), mu[1] ≈ 0.35 (low scores)
    """
    A = np.asarray(A, dtype=np.float64)

    if A.size == 0:
        return np.array([])

    # Vectorized computation: mean across columns (axis=1) for each row
    mu = np.mean(A, axis=1)

    return np.clip(mu, 0.0, 1.0)


def ssd_score(mu: np.ndarray, risk_aversion: float = 5.0) -> float:
    """
    Compute Second-Order Stochastic Dominance (SSD) score using exponential utility.

    Implements the certainty equivalent formula from stochastic dominance theory:
        μ_SSD = -(1/λ) · ln(E[e^(-λX)])

    Where:
        - λ (lambda): Risk aversion parameter (higher = more risk-averse)
        - X: Random variable representing metric scores
        - E[·]: Expected value (sample mean)

    This formula represents the certainty equivalent of an exponential utility
    function U(x) = -e^(-λx), which exhibits constant absolute risk aversion (CARA).

    Interpretation:
        - λ = 0: Risk-neutral (reduces to arithmetic mean)
        - λ = 5-7: Moderate risk aversion (recommended for aggregation)
        - λ = 10+: High risk aversion (heavily penalizes poor scores)

    Properties:
        - μ_SSD ∈ [min(mu), mean(mu)]: Properly bounded
        - Higher λ → result closer to min (worst case)
        - Lower λ → result closer to mean (average case)
        - Mathematically rigorous (from utility theory)

    Args:
        mu: Array of scores to aggregate, values in [0, 1]
        risk_aversion: Risk aversion parameter λ (default: 5.0)

    Returns:
        SSD-aggregated score in [0, 1]

    References:
        - Levy, H. (2016). Stochastic Dominance: Investment Decision Making
        - Pratt, J. W. (1964). Risk Aversion in the Small and in the Large

    Example:
        >>> mu = np.array([0.8, 0.6, 0.9])
        >>> ssd_score(mu, risk_aversion=5.0)  # ≈ 0.74 (below mean of 0.77)
        >>> ssd_score(mu, risk_aversion=10.0)  # ≈ 0.68 (more risk-averse)
    """
    mu = np.asarray(mu, dtype=np.float64)
    n = len(mu)

    if n == 0:
        return 0.0

    if n == 1:
        return float(mu[0])

    # Handle edge cases where all values are identical
    if np.allclose(mu, mu[0]):
        return float(mu[0])

    # Exponential utility certainty equivalent formula
    # CE = -(1/λ) · ln(E[e^(-λX)])
    lambda_param = risk_aversion

    # Compute exponential utilities: e^(-λx_i)
    exp_utilities = np.exp(-lambda_param * mu)

    # Compute expected value: E[e^(-λX)] = (1/n) Σ e^(-λx_i)
    mean_exp_utility = np.mean(exp_utilities)

    # Apply certainty equivalent formula
    ce = -np.log(mean_exp_utility) / lambda_param

    # Clip to valid range for numerical stability
    # Theoretical bounds: min(mu) ≤ CE ≤ mean(mu)
    ce = np.clip(ce, np.min(mu), np.mean(mu))

    return float(ce)


def aggregate_metrics(
    A: np.ndarray,
    normalize: bool = False,
    lower_is_better: np.ndarray | None = None,
    risk_aversion: float = 5.0,
) -> tuple[np.ndarray, float]:
    """
    Full pipeline: Collapse metrics matrix A into global score Q.

    Pipeline:
        1. (Optional) Normalize A to [0, 1] using robust percentiles
        2. Compute FSD scores μ for each variable (row)
        3. Compute SSD score Q from μ using exponential utility

    Args:
        A: Metrics matrix of shape (n, m)
           - n = number of variables (e.g., columns in dataset)
           - m = number of metrics (e.g., KS, Wasserstein, etc.)
        normalize: If True, apply normalize_metrics first
        lower_is_better: Boolean array for normalization (if normalize=True)
        risk_aversion: Risk aversion parameter λ for SSD (default: 5.0)

    Returns:
        tuple (mu, Q) where:
            - mu: Array of shape (n,) with FSD scores per variable
            - Q: Global SSD-based score in [0, 1]

    Example:
        >>> A = np.array(
        ...     [
        ...         [0.9, 0.8, 0.85, 0.7, 0.95],  # Variable 1: mostly good
        ...         [0.6, 0.5, 0.55, 0.4, 0.65],  # Variable 2: moderate
        ...         [0.3, 0.2, 0.25, 0.1, 0.35],  # Variable 3: poor
        ...         [0.8, 0.9, 0.85, 0.75, 0.8],  # Variable 4: good
        ...     ]
        ... )
        >>> mu, Q = aggregate_metrics(A, risk_aversion=7.0)
        >>> print(f"Variable scores: {mu}")
        >>> print(f"Global score Q: {Q:.4f}")
    """
    A = np.asarray(A, dtype=np.float64)

    # Step 1: Normalize if requested
    if normalize:
        A = normalize_metrics(A, lower_is_better=lower_is_better)

    # Step 2: Compute FSD scores per variable
    mu = fsd_scores(A)

    # Step 3: Compute global SSD score
    Q = ssd_score(mu, risk_aversion)

    return mu, Q


# =============================================================================
# Hierarchical Aggregation Functions
# =============================================================================
# Note: fsd_aggregate and ssd_aggregate have been removed as they were simple
# wrappers around fsd_score_for_row and ssd_score. Use those functions directly.


def hierarchical_aggregate(
    scores: np.ndarray,
    level: AggregationLevel,
    risk_aversion: float = 5.0,
) -> float:
    """
    Aggregate scores at a specific hierarchy level.

    Uses FSD for levels 1-2 (less penalizing, captures average behavior)
    and SSD for levels 3-4 (risk-averse, penalizes poor performance).

    Args:
        scores: Array of scores to aggregate, values in [0, 1]
        level: Aggregation level determining the method (FSD or SSD)
        risk_aversion: Risk aversion parameter λ for SSD (default: 5.0)

    Returns:
        Aggregated score in [0, 1]

    Example:
        >>> scores = np.array([0.9, 0.8, 0.85, 0.7])
        >>> # Level 1-2: Use FSD (less penalizing)
        >>> fsd_result = hierarchical_aggregate(scores, AggregationLevel.LEVEL_1_METRICS)
        >>> # Level 3-4: Use SSD (risk-averse)
        >>> ssd_result = hierarchical_aggregate(
        ...     scores, AggregationLevel.LEVEL_3_CATEGORY, risk_aversion=7.0
        ... )
    """
    scores = np.asarray(scores, dtype=np.float64)

    if len(scores) == 0:
        return 0.0

    # FSD for levels 1-2 (mean-based), SSD for levels 3-4 (risk-averse)
    if level in (
        AggregationLevel.LEVEL_1_METRICS,
        AggregationLevel.LEVEL_2_SUBCATEGORY,
    ):
        return fsd_score_for_row(scores)
    return ssd_score(scores, risk_aversion)


def _validate_weights(weights: np.ndarray, n: int) -> np.ndarray:
    """Validate and normalize a non-negative, non-zero weight vector of length n."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n,):
        raise ValueError(f"weights shape {weights.shape} != expected ({n},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("sum of weights must be > 0")
    return weights / total


def weighted_fsd_score(scores: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted FSD score: weighted arithmetic mean of scores in [0, 1]."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return 0.0
    if weights is None:
        return float(np.clip(np.mean(scores), 0.0, 1.0))
    w = _validate_weights(weights, n)
    return float(np.clip(np.sum(w * scores), 0.0, 1.0))


def weighted_ssd_score(
    scores: np.ndarray,
    weights: np.ndarray | None = None,
    risk_aversion: float = 5.0,
) -> float:
    """Weighted SSD score using exponential utility certainty equivalent.

    CE = -(1/lambda) * ln(sum_i w_i * exp(-lambda * x_i)) with weights normalized to sum 1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return 0.0
    if n == 1:
        return float(scores[0])
    if np.allclose(scores, scores[0]):
        return float(scores[0])
    w = _validate_weights(weights, n) if weights is not None else np.full(n, 1.0 / n)
    exp_utilities = np.exp(-risk_aversion * scores)
    mean_exp_utility = float(np.sum(w * exp_utilities))
    ce = -np.log(mean_exp_utility) / risk_aversion
    weighted_mean = float(np.sum(w * scores))
    return float(np.clip(ce, float(np.min(scores)), weighted_mean))


def weighted_hierarchical_aggregate(
    scores: np.ndarray,
    weights: np.ndarray | None,
    level: AggregationLevel,
    risk_aversion: float = 5.0,
) -> float:
    """Weighted variant of hierarchical_aggregate honoring per-score weights."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    if level in (
        AggregationLevel.LEVEL_1_METRICS,
        AggregationLevel.LEVEL_2_SUBCATEGORY,
    ):
        return weighted_fsd_score(scores, weights)
    return weighted_ssd_score(scores, weights, risk_aversion)


def aggregate_hierarchical(
    level1_scores: dict,
    risk_aversion: float = 5.0,
) -> tuple[dict, float]:
    """
    Full hierarchical aggregation from level 1 to final score.

    This function performs multi-level aggregation:
        Level 1→2: Aggregate metric scores within each category (FSD)
        Level 2→3: Aggregate category scores to final (SSD)

    Args:
        level1_scores: Dictionary mapping category names to arrays of metric scores.
                      Each array contains individual metric scores in [0, 1].
                      Example: {"tails": [0.9, 0.8, 0.7], "scales": [0.85, 0.75]}
        risk_aversion: Risk aversion parameter λ for SSD aggregation (default: 5.0)

    Returns:
        tuple of (category_scores, final_score) where:
            - category_scores: dict mapping category names to their FSD-aggregated scores
            - final_score: Final SSD-aggregated validation score

    Example:
        >>> scores = {
        ...     "tails": np.array([0.9, 0.8, 0.85]),
        ...     "scales": np.array([0.7, 0.75, 0.72]),
        ...     "coverage": np.array([0.95, 0.92, 0.88]),
        ... }
        >>> category_scores, final = aggregate_hierarchical(scores, risk_aversion=7.0)
        >>> # category_scores: {'tails': 0.85, 'scales': 0.72, 'coverage': 0.92}
        >>> # final: SSD aggregation of [0.85, 0.72, 0.92] with λ=7.0
    """
    # Level 2: Aggregate each category using FSD
    category_scores = {}
    for category, scores in level1_scores.items():
        scores_arr = np.asarray(scores, dtype=np.float64)
        if len(scores_arr) == 0:
            category_scores[category] = 0.0
        else:
            category_scores[category] = hierarchical_aggregate(
                scores_arr,
                AggregationLevel.LEVEL_2_SUBCATEGORY,
                risk_aversion,
            )

    # Level 3/4: Aggregate categories using SSD
    all_category_scores = np.array(list(category_scores.values()))
    if len(all_category_scores) == 0:
        final_score = 0.0
    else:
        final_score = hierarchical_aggregate(
            all_category_scores,
            AggregationLevel.LEVEL_3_CATEGORY,
            risk_aversion,
        )

    return category_scores, final_score


def aggregate_domains(
    domain_scores: dict,
) -> tuple[dict, float]:
    """
    Aggregate domain scores (Fidelity, Privacy, Utility) into final validation score.

    Uses SSD (Level 4) for risk-averse aggregation that ensures poor
    performance in any domain significantly impacts the final score.

    Args:
        domain_scores: Dictionary mapping domain names to their scores.
                      Example: {"fidelity": 0.85, "privacy": 0.72, "utility": 0.90}

    Returns:
        tuple of (domain_scores, final_validation_score) where:
            - domain_scores: Same as input (for consistency)
            - final_validation_score: SSD-aggregated final score

    Example:
        >>> domains = {"fidelity": 0.85, "privacy": 0.72, "utility": 0.90}
        >>> _, final = aggregate_domains(domains)
        >>> # final uses SSD to penalize the lower privacy score
    """
    scores_arr = np.array(list(domain_scores.values()), dtype=np.float64)

    if len(scores_arr) == 0:
        return domain_scores, 0.0

    final_score = hierarchical_aggregate(scores_arr, AggregationLevel.LEVEL_4_FINAL)

    return domain_scores, final_score
