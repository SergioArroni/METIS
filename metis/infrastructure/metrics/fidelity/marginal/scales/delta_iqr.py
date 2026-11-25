"""Delta IQR metric for scale distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric, relative_difference


@register("fidelity.delta_iqr")
class DeltaIQRMetric(MarginalMetric):
    """
    Delta IQR - compare interquartile ranges between distributions.

    ``|IQR_real - IQR_synth| / IQR_real``: the relative difference in IQR
    (spread of the middle 50% of values). When the real IQR is 0
    (zero-inflated columns) the difference is expressed in real standard
    deviations instead.

    IQR is a robust measure of scale, resistant to outliers.
    """

    name: str = "delta_iqr"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """Relative IQR difference in [0, inf), lower is better."""
        real_iqr = float(np.subtract(*np.percentile(real_col, [75, 25])))
        synth_iqr = float(np.subtract(*np.percentile(synth_col, [75, 25])))
        return relative_difference(real_iqr - synth_iqr, real_iqr, real_col.std())
