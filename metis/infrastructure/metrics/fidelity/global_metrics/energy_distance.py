"""
Energy Distance metric (multivariate).

Energy Distance is a statistical distance between probability distributions
based on the concept of potential energy. It generalizes the Cramér distance
to multivariate settings.
"""

import numpy as np

from ...registry import register
from ..fidelity_base import GlobalFidelityMetric


@register("fidelity.energy_distance")
class EnergyDistanceMetric(GlobalFidelityMetric):
    """
    Multivariate Energy Distance between real and synthetic distributions.

    Energy distance is defined as:
    E(P,Q) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]

    where X,X' ~ P and Y,Y' ~ Q are independent samples.

    Properties:
    - Non-negative, equals 0 iff P = Q
    - Does not require density estimation
    - Works well in high dimensions
    - More sensitive to differences in tails than MMD

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> real = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])
        >>> synth = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])
        >>> metric = EnergyDistanceMetric()
        >>> metric.fit(real, synth)
        >>> result = metric.compute()
        >>> print(f"Energy Distance Score: {result.value:.4f}")
    """

    name: str = "energy_distance"

    def __init__(
        self,
        max_samples: int = 2000,
        alpha: float = 1.0,
    ):
        """
        Initialize the Energy Distance metric.

        Args:
            max_samples: Maximum samples to use (for computational efficiency)
            alpha: Exponent for distance (default 1.0, can use 0 < alpha < 2)
        """
        super().__init__()
        self.max_samples = max_samples
        self.alpha = alpha

    def _compute_global(self) -> tuple[float, float, dict]:
        """
        Compute Energy Distance between real and synthetic data.

        Returns:
            tuple of (raw_value, normalized_value, details)
        """
        # Get numeric columns
        real_num = self._real_data.select_dtypes(include=[np.number])
        synth_num = self._synth_data.select_dtypes(include=[np.number])
        common_cols = list(set(real_num.columns) & set(synth_num.columns))

        if not common_cols:
            raise ValueError("No common numeric columns")

        # Prepare data
        X = real_num[common_cols].dropna().values
        Y = synth_num[common_cols].dropna().values

        # Fair comparison: use same sample size for both datasets
        n_samples = min(len(X), len(Y), self.max_samples)

        # Get seed from context for reproducibility
        seed = self._context.get("seed", 42)
        rng = np.random.default_rng(seed)

        if len(X) > n_samples:
            idx = rng.choice(len(X), n_samples, replace=False)
            X = X[idx]
        if len(Y) > n_samples:
            idx = rng.choice(len(Y), n_samples, replace=False)
            Y = Y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-10
        X = (X - mean) / std
        Y = (Y - mean) / std

        # Compute Energy Distance
        energy_dist = self._compute_energy_distance(X, Y)

        # Normalize to [0, 1] where 1 = similar
        # Energy distance can be large, normalize using empirical bounds
        # For standardized data, typical values are in [0, 4]
        normalized = 1.0 - min(energy_dist / 4.0, 1.0)

        details = {
            "raw_energy_distance": energy_dist,
            "alpha": self.alpha,
            "n_real_samples": len(X),
            "n_synth_samples": len(Y),
            "n_features": len(common_cols),
        }

        return energy_dist, normalized, details

    def _compute_energy_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute energy distance using the V-statistic estimator.

        E(X,Y) = 2*E[||X-Y||^α] - E[||X-X'||^α] - E[||Y-Y'||^α]
        """
        m, n = len(X), len(Y)

        if m < 2 or n < 2:
            return 0.0

        # E[||X-Y||^α] - cross distances
        E_XY = self._mean_distances(X, Y)

        # E[||X-X'||^α] - within X distances
        E_XX = self._mean_distances_within(X)

        # E[||Y-Y'||^α] - within Y distances
        E_YY = self._mean_distances_within(Y)

        energy = 2 * E_XY - E_XX - E_YY
        return float(max(0, energy))

    def _mean_distances(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute mean distance between samples from X and Y."""
        # For efficiency, use sampling if matrices are large
        m, n = len(X), len(Y)

        if m * n > 1e7:  # Sample if too many pairs
            n_samples = int(np.sqrt(1e7))
            seed = self._context.get("seed", 42)
            rng = np.random.default_rng(seed)
            X_idx = rng.choice(m, min(n_samples, m), replace=False)
            Y_idx = rng.choice(n, min(n_samples, n), replace=False)
            X = X[X_idx]
            Y = Y[Y_idx]

        # Compute ||X_i - Y_j||^α for all pairs
        # ||x-y||² = ||x||² + ||y||² - 2x·y
        X_sqnorm = np.sum(X**2, axis=1, keepdims=True)
        Y_sqnorm = np.sum(Y**2, axis=1, keepdims=True)
        sq_dist = X_sqnorm + Y_sqnorm.T - 2 * (X @ Y.T)
        sq_dist = np.maximum(sq_dist, 0)  # Numerical stability

        if self.alpha == 1.0:
            distances = np.sqrt(sq_dist)
        elif self.alpha == 2.0:
            distances = sq_dist
        else:
            distances = np.power(sq_dist, self.alpha / 2)

        return float(distances.mean())

    def _mean_distances_within(self, X: np.ndarray) -> float:
        """Compute mean distance within samples of X."""
        m = len(X)

        if m < 2:
            return 0.0

        if m * m > 1e7:  # Sample if too many pairs
            n_samples = int(np.sqrt(1e7))
            seed = self._context.get("seed", 42)
            rng = np.random.default_rng(seed)
            idx = rng.choice(m, min(n_samples, m), replace=False)
            X = X[idx]
            m = len(X)

        # Compute ||X_i - X_j||^α for all pairs
        X_sqnorm = np.sum(X**2, axis=1, keepdims=True)
        sq_dist = X_sqnorm + X_sqnorm.T - 2 * (X @ X.T)
        sq_dist = np.maximum(sq_dist, 0)

        if self.alpha == 1.0:
            distances = np.sqrt(sq_dist)
        elif self.alpha == 2.0:
            distances = sq_dist
        else:
            distances = np.power(sq_dist, self.alpha / 2)

        # Exclude diagonal (self-distances = 0)
        np.fill_diagonal(distances, 0)

        # Mean over all pairs (excluding diagonal)
        return float(distances.sum() / (m * (m - 1)))
