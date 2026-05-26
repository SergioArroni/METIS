"""Gini coefficient delta metric for coverage distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric


@register("fidelity.gini_delta")
class GiniDeltaMetric(MarginalMetric):
    """
    Delta of Gini coefficient between distributions.

    The Gini coefficient measures inequality in a distribution.
    Values range from 0 (perfect equality) to 1 (maximum inequality).

    This metric computes the absolute difference in Gini coefficients
    between real and synthetic data.

    Note: Only works for numeric columns.
    """

    name: str = "gini_delta"
    is_distance: bool = True

    def _compute_gini(self, values: np.ndarray) -> float:
        """
        Compute Gini coefficient.

        Args:
            values: Array of values

        Returns:
            Gini coefficient in [0, 1]
        """
        if len(values) == 0:
            return 0.0

        # Shift values to be non-negative
        shifted = values - values.min() + 1
        n = len(shifted)

        sorted_vals = np.sort(shifted)
        cumulative = np.cumsum(sorted_vals)

        # Gini = (2 * sum(i * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
        numerator = 2 * np.sum(np.arange(1, n + 1) * sorted_vals) - (n + 1) * cumulative[-1]
        denominator = n * cumulative[-1]

        if denominator > 1e-10:
            return numerator / denominator
        return 0.0

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Gini Delta for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Absolute Gini difference in [0, 1], lower is better
        """
        gini_real = self._compute_gini(real_col.values)
        gini_synth = self._compute_gini(synth_col.values)

        return abs(gini_real - gini_synth)
