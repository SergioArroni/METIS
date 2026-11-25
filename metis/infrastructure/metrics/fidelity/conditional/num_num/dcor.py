"""
Distance Correlation (dCor) metric for Num↔Num pairs.

Measures non-linear dependency preservation between numeric column pairs.
"""

import warnings

import numpy as np
import pandas as pd

from ....registry import register
from ...fidelity_base import NumNumPairMetric


@register("fidelity.dcor")
class DistanceCorrelationMetric(NumNumPairMetric):
    """
    Distance Correlation preservation metric.

    Measures how well synthetic data preserves non-linear dependencies
    between pairs of numeric columns. Distance correlation equals zero
    if and only if the variables are independent.

    The bias-corrected distance correlation of Székely and Rizzo (2014, Ann.
    Stat. 42(6)) is used: U-centred distance matrices (zero diagonal) instead
    of double-centred ones.  The plain estimate is biased upwards by an
    amount that shrinks with n (two independent columns give dCor ≈ 0.1 with
    500 rows), so real and synthetic tables of different sizes, or with gaps
    in different columns, compared different biases; the corrected one is ≈ 0
    for independent columns at any n.  A negative corrected dCor² is
    reported as 0.

    Note: Uses subsampling for large datasets to avoid O(n²) memory explosion.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 4, 9, 16, 25]})
        >>> synth = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 5, 10, 17, 26]})
        >>> metric = DistanceCorrelationMetric()
        >>> metric.fit(real, synth)
        >>> result = metric.compute()
    """

    name: str = "dcor"
    max_samples: int = 2000  # Limit to avoid O(n²) memory explosion

    # Memory budget for the per-column centred distance matrices kept between
    # pairs (each is n x n float64). Columns beyond the budget are recomputed.
    cache_budget_bytes: int = 512 * 1024 * 1024

    @staticmethod
    def _double_centered(x: np.ndarray) -> np.ndarray:
        """U-centred distance matrix (Székely and Rizzo 2014)::

            Ã_ij = a_ij − a_i./(n−2) − a_.j/(n−2) + a_../((n−1)(n−2)),  Ã_ii = 0

        with the row, column and total *sums* of ``a_ij = |x_i − x_j|``.
        """
        n = len(x)
        a = np.abs(x[:, None] - x[None, :])
        row = a.sum(axis=1, keepdims=True) / (n - 2)
        col = a.sum(axis=0, keepdims=True) / (n - 2)
        u = a - row - col + a.sum() / ((n - 1) * (n - 2))
        np.fill_diagonal(u, 0.0)
        return u

    def _subsample_index(self, n: int) -> np.ndarray | None:
        """Common subsample (same seed, same n -> same rows for every column)."""
        if n <= self.max_samples:
            return None
        rng = np.random.default_rng(self.seed)
        return rng.choice(n, self.max_samples, replace=False)

    def _centered_for(self, side: str, col: pd.Series, full: bool) -> tuple[np.ndarray, float]:
        """Centred matrix and distance variance of ``col``, cached when complete.

        ``full`` means the series is the whole column (no rows dropped for this
        pair), so the matrix is the same for every pair the column is in.
        """
        key = (side, col.name)
        if full and key in self._centered_cache:
            return self._centered_cache[key]
        x = col.to_numpy(dtype=float)
        idx = self._subsample_index(len(x))
        if idx is not None:
            x = x[idx]
        centred = self._double_centered(x)
        entry = (centred, float((centred * centred).mean()))
        if full and self._cache_bytes + centred.nbytes <= self.cache_budget_bytes:
            self._centered_cache[key] = entry
            self._cache_bytes += centred.nbytes
        return entry

    def _prepare_pairs(self) -> None:
        self._centered_cache: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
        self._cache_bytes = 0
        self._n_rows = {"real": len(self._real_data), "synth": len(self._synth_data)}

    def _dcor_side(self, side: str, x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 4:
            return 0.0
        full = n == self._n_rows[side]
        a_mat, dvar_x = self._centered_for(side, x, full)
        b_mat, dvar_y = self._centered_for(side, y, full)
        if dvar_x * dvar_y <= 0:
            return 0.0
        dcov_sq = float((a_mat * b_mat).mean())
        return float(np.sqrt(max(dcov_sq, 0.0) / np.sqrt(dvar_x * dvar_y)))

    def _compute_dcor(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance correlation between two numeric arrays.

        Distance correlation equals zero if and only if the variables
        are independent. Unlike Pearson, it detects non-linear relationships.

        Args:
            x: First numeric array
            y: Second numeric array

        Returns:
            Distance correlation value in [0, 1]
        """
        n = len(x)
        if n < 4:
            return 0.0

        # Subsample if too large (O(n²) memory complexity)
        if n > self.max_samples:
            seed = self.seed
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, self.max_samples, replace=False)
            x = x[idx]
            y = y[idx]
            n = self.max_samples

        # U-centred distance matrices (n×n), bias-corrected
        A = self._double_centered(np.asarray(x, dtype=float))
        B = self._double_centered(np.asarray(y, dtype=float))

        # Distance covariance and variances
        dcov_sq = (A * B).mean()
        dvar_x = (A * A).mean()
        dvar_y = (B * B).mean()

        if dvar_x * dvar_y <= 0:
            return 0.0

        return float(np.sqrt(max(dcov_sq, 0.0) / np.sqrt(dvar_x * dvar_y)))

    def _compute_pair(
        self,
        real_col1: pd.Series,
        real_col2: pd.Series,
        synth_col1: pd.Series,
        synth_col2: pd.Series,
    ) -> tuple[float, float]:
        """Compute Distance Correlation for a column pair."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if getattr(self, "_centered_cache", None) is None:
                real_val = self._compute_dcor(real_col1.values, real_col2.values)
                synth_val = self._compute_dcor(synth_col1.values, synth_col2.values)
            else:
                real_val = self._dcor_side("real", real_col1, real_col2)
                synth_val = self._dcor_side("synth", synth_col1, synth_col2)

        return real_val, synth_val
