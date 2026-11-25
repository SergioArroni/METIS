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

    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """Gini coefficient ``sum_u sum_v |x_u - x_v| / (2 n^2 mean(x))``.

        NaN when the mean is not positive. Invariant to the unit of the
        column: ``G(c * x) = G(x)``.
        """
        values = np.sort(np.asarray(values, dtype=float))
        n = len(values)
        if n == 0:
            return float("nan")
        total = values.sum()
        if total <= 1e-12 * max(1.0, float(np.abs(values).sum())):
            return float("nan")
        # sum_u sum_v |x_u - x_v| = 2 sum_i (2i - n - 1) x_(i) for sorted x.
        ranks = np.arange(1, n + 1)
        return float(np.sum((2 * ranks - n - 1) * values) / (n * total))

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Gini Delta for a single column.

        The Gini coefficient is defined for non-negative quantities. Columns
        whose real values are all non-negative use the raw values (as in the
        paper). Otherwise both tables are shifted by the real minimum, a
        shift expressed in the column's own units, so the result does not
        depend on them. It used to be ``x - min(x) + 1``: the ``+ 1`` is one
        unit (euros or cents gave different Ginis) and each table used its
        own minimum.

        Returns:
            ``|G_real - G_synth|`` clipped to [0, 1], lower is better
        """
        real = real_col.to_numpy(dtype=float)
        synth = synth_col.to_numpy(dtype=float)
        shift = min(0.0, float(real.min()))
        gini_real = self._compute_gini(real - shift)
        gini_synth = self._compute_gini(synth - shift)
        return float(min(1.0, abs(gini_real - gini_synth)))
