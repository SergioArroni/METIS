"""
Distribution utilities for marginal metrics.

Provides helper functions for converting data to probability distributions
and aligning distributions for comparison.
"""

import numpy as np
import pandas as pd


def get_distribution(series: pd.Series, n_bins: int = 50) -> np.ndarray:
    """
    Convert a data series to a probability distribution.

    For categorical data (object/category dtype or <20 unique values),
    returns normalized value counts.
    For numeric data, returns a histogram-based probability distribution.

    Args:
        series: Input data series
        n_bins: Number of bins for numeric data histograms

    Returns:
        Probability distribution as numpy array (sums to 1)

    Example:
        >>> import pandas as pd
        >>> data = pd.Series([1, 2, 2, 3, 3, 3])
        >>> dist = get_distribution(data)
        >>> dist.sum()  # Should be ~1.0
    """
    if series.dtype in ["object", "category"] or series.nunique() < 20:
        # Categorical: use value counts
        counts = series.value_counts(normalize=True, dropna=True)
        return counts.values
    # Numeric: bin into histogram
    hist, _ = np.histogram(series.dropna(), bins=n_bins, density=True)
    # Normalize to sum to 1
    return hist / (hist.sum() + 1e-10)


def align_distributions(
    real: pd.Series, synth: pd.Series, n_bins: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two distributions for comparison, handling missing categories.

    For categorical data, ensures both distributions have the same categories.
    For numeric data, uses common bin edges for both histograms.

    Adds a small epsilon for numerical stability and renormalizes.

    Args:
        real: Real data series
        synth: Synthetic data series
        n_bins: Number of bins for numeric data

    Returns:
        tuple of (real_distribution, synth_distribution) as numpy arrays,
        both normalized to sum to 1.

    Example:
        >>> import pandas as pd
        >>> real = pd.Series(["A", "A", "B", "C"])
        >>> synth = pd.Series(["A", "B", "B", "D"])
        >>> p, q = align_distributions(real, synth)
        >>> len(p) == len(q)  # Same length
        True
    """
    if real.dtype in ["object", "category"] or real.nunique() < 20:
        # Categorical: align on all categories
        all_categories = set(real.dropna().unique()) | set(synth.dropna().unique())

        real_counts = real.value_counts(normalize=True, dropna=True)
        synth_counts = synth.value_counts(normalize=True, dropna=True)

        p = np.array([real_counts.get(c, 0) for c in all_categories])
        q = np.array([synth_counts.get(c, 0) for c in all_categories])
    else:
        # Numeric: use common bins
        combined = pd.concat([real.dropna(), synth.dropna()])
        bins = np.histogram_bin_edges(combined, bins=n_bins)

        p, _ = np.histogram(real.dropna(), bins=bins, density=False)
        q, _ = np.histogram(synth.dropna(), bins=bins, density=False)

        # Normalize
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)

    # Add small epsilon for numerical stability
    eps = 1e-10
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    # Renormalize
    p = p / p.sum()
    q = q / q.sum()

    return p, q


def compute_histogram(
    series: pd.Series, bins: int = 50, density: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram for a numeric series.

    Args:
        series: Numeric data series
        bins: Number of bins
        density: If True, normalize to probability density

    Returns:
        tuple of (histogram_values, bin_edges)
    """
    clean_data = series.dropna()
    hist, edges = np.histogram(clean_data, bins=bins, density=density)
    return hist, edges


def compute_empirical_cdf(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical cumulative distribution function.

    Args:
        series: Data series

    Returns:
        tuple of (sorted_values, cdf_values)
    """
    clean_data = series.dropna().values
    sorted_data = np.sort(clean_data)
    n = len(sorted_data)
    cdf = np.arange(1, n + 1) / n
    return sorted_data, cdf
