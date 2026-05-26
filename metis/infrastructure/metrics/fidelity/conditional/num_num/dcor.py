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
            seed = self._context.get("seed", 42)
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, self.max_samples, replace=False)
            x = x[idx]
            y = y[idx]
            n = self.max_samples

        # Distance matrices (n×n)
        a = np.abs(x[:, None] - x)
        b = np.abs(y[:, None] - y)

        # Double centering
        a_row = a.mean(axis=1, keepdims=True)
        a_col = a.mean(axis=0, keepdims=True)
        a_mean = a.mean()
        A = a - a_row - a_col + a_mean

        b_row = b.mean(axis=1, keepdims=True)
        b_col = b.mean(axis=0, keepdims=True)
        b_mean = b.mean()
        B = b - b_row - b_col + b_mean

        # Distance covariance and variances
        dcov_sq = (A * B).mean()
        dvar_x = (A * A).mean()
        dvar_y = (B * B).mean()

        if dvar_x * dvar_y <= 0:
            return 0.0

        return float(np.sqrt(dcov_sq / np.sqrt(dvar_x * dvar_y)))

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
            real_val = self._compute_dcor(real_col1.values, real_col2.values)
            synth_val = self._compute_dcor(synth_col1.values, synth_col2.values)

        return real_val, synth_val
