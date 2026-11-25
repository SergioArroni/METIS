"""Delta Median Absolute Deviation metric for scale distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric, relative_difference


@register("fidelity.delta_mad")
class DeltaMADMetric(MarginalMetric):
    """
    Delta Median Absolute Deviation - compare MAD between distributions.

    ``|MAD_real - MAD_synth| / MAD_real``: the relative difference in MAD, a
    robust measure of statistical dispersion. When the real MAD is 0
    (zero-inflated columns) the difference is expressed in real standard
    deviations instead.
    """

    name: str = "delta_mad"
    is_distance: bool = True

    @staticmethod
    def _mad(values: pd.Series) -> float:
        return float(np.median(np.abs(values - np.median(values))))

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """Relative MAD difference in [0, inf), lower is better."""
        real_mad = self._mad(real_col)
        return relative_difference(real_mad - self._mad(synth_col), real_mad, real_col.std())
