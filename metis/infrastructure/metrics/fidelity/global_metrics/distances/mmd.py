"""
Maximum Mean Discrepancy (MMD) metric.

MMD is a kernel-based distance between probability distributions.
It measures the difference in mean embeddings of two distributions
in a reproducing kernel Hilbert space (RKHS).
"""

import numpy as np

from metis.shared.normalization import normalize_metric_value

from ....registry import register
from ...fidelity_base import GlobalFidelityMetric, GlobalMetricNotApplicable, standardise_with_missingness


@register("fidelity.mmd")
class MMDMetric(GlobalFidelityMetric):
    """
    Maximum Mean Discrepancy between real and synthetic distributions.

    MMD uses kernel functions to compare distributions without assuming
    a parametric form. Commonly used kernels:
    - RBF (Gaussian): Captures smooth similarities
    - Linear: Compares means
    - Polynomial: Captures higher-order moments

    The metric is computed using an unbiased estimator:
    MMD² = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> real = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])
        >>> synth = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])
        >>> metric = MMDMetric()
        >>> metric.fit(real, synth)
        >>> result = metric.compute()
        >>> print(f"MMD Score: {result.value:.4f}")
    """

    name: str = "mmd"

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float | None = None,
        max_samples: int = 2000,
    ):
        """
        Initialize the MMD metric.

        Args:
            kernel: Kernel type ("rbf", "linear", "polynomial")
            gamma: RBF kernel bandwidth. If None, uses median heuristic
            max_samples: Maximum samples to use (for computational efficiency)
        """
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.max_samples = max_samples

    def _compute_global(self) -> tuple[float, float, dict]:
        """
        Compute MMD between real and synthetic data.

        Returns:
            tuple of (raw_value, normalized_value, details)
        """
        # Common numeric columns (sorted); an ordinal enters by its rank
        # ``__ord`` (its categorical twin is not numeric, so it counts once).
        common_cols = self._get_numeric_columns(include_ordinal_ranks=True)
        real_num, synth_num = self._real_data, self._synth_data

        if not common_cols:
            raise GlobalMetricNotApplicable("No common numeric columns")

        # Prepare data
        # Every row counts; gaps are encoded, not dropped (see
        # ``standardise_with_missingness``).
        X = real_num[common_cols].to_numpy(dtype=float)
        Y = synth_num[common_cols].to_numpy(dtype=float)

        # Fair comparison: use same sample size for both datasets
        n_samples = min(len(X), len(Y), self.max_samples)

        # Get seed from context for reproducibility
        seed = self.seed
        rng = np.random.default_rng(seed)

        if len(X) > n_samples:
            idx = rng.choice(len(X), n_samples, replace=False)
            X = X[idx]
        if len(Y) > n_samples:
            idx = rng.choice(len(Y), n_samples, replace=False)
            Y = Y[idx]

        # Standardize with the real statistics; gaps -> real mean + indicator
        X, Y, n_indicators = standardise_with_missingness(X, Y)

        # Compute MMD
        mmd_value = self._compute_mmd(X, Y)

        # Normalize to [0, 1] where 1 = similar, with the central registry
        # (``fidelity.mmd`` -> exp(-MMD / scale)) like every other metric.
        normalized = normalize_metric_value(self.metric_id, mmd_value)

        details = {
            "raw_mmd": mmd_value,
            "kernel": self.kernel,
            "gamma": self.gamma if self.gamma else "median_heuristic",
            "n_real_samples": len(X),
            "n_synth_samples": len(Y),
            "n_features": len(common_cols),
            "n_missingness_indicators": n_indicators,
        }

        return mmd_value, normalized, details

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute MMD using unbiased estimator.

        MMD² = 1/(m(m-1)) Σ k(xi,xj) + 1/(n(n-1)) Σ k(yi,yj) - 2/(mn) Σ k(xi,yj)
        """
        m, n = len(X), len(Y)

        if m < 2 or n < 2:
            return 0.0

        # Compute gamma if using RBF and not specified
        gamma = self.gamma
        if self.kernel == "rbf" and gamma is None:
            gamma = self._median_heuristic(X, Y)

        # Compute kernel matrices
        K_XX = self._compute_kernel(X, X, gamma)
        K_YY = self._compute_kernel(Y, Y, gamma)
        K_XY = self._compute_kernel(X, Y, gamma)

        # Unbiased MMD estimator
        # Remove diagonal for XX and YY (self-comparisons)
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)

        mmd_sq = K_XX.sum() / (m * (m - 1)) + K_YY.sum() / (n * (n - 1)) - 2 * K_XY.sum() / (m * n)

        # Return sqrt for MMD (not MMD²)
        return float(np.sqrt(max(0, mmd_sq)))

    def _compute_kernel(
        self, X: np.ndarray, Y: np.ndarray, gamma: float | None = None
    ) -> np.ndarray:
        """Compute kernel matrix between X and Y."""
        if self.kernel == "linear":
            return X @ Y.T
        if self.kernel == "polynomial":
            return (X @ Y.T + 1) ** 3
        # rbf
        # ||x - y||² = ||x||² + ||y||² - 2x·y
        X_sqnorm = np.sum(X**2, axis=1, keepdims=True)
        Y_sqnorm = np.sum(Y**2, axis=1, keepdims=True)
        sq_dist = X_sqnorm + Y_sqnorm.T - 2 * (X @ Y.T)
        return np.exp(-gamma * sq_dist)

    def _median_heuristic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute gamma using median heuristic."""
        # Sample distances for efficiency
        n_samples = min(500, len(X), len(Y))
        seed = self.seed
        rng = np.random.default_rng(seed)
        X_sample = X[rng.choice(len(X), n_samples, replace=False)]
        Y_sample = Y[rng.choice(len(Y), n_samples, replace=False)]

        # Compute pairwise distances
        combined = np.vstack([X_sample, Y_sample])
        sq_dist = (
            np.sum(combined**2, axis=1, keepdims=True)
            + np.sum(combined**2, axis=1, keepdims=True).T
            - 2 * (combined @ combined.T)
        )

        # Median of non-zero distances
        sq_dist_flat = sq_dist[np.triu_indices_from(sq_dist, k=1)]
        median_dist = np.median(sq_dist_flat[sq_dist_flat > 0])

        # Bandwidth parameter for RBF kernel
        return 1.0 / (2 * median_dist + 1e-10)
