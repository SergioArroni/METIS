"""Delta Median metric for scale distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric, relative_difference


@register("fidelity.delta_median")
class DeltaMedianMetric(MarginalMetric):
    """
    Delta Median - compare medians between distributions.

    ``|median_real - median_synth| / IQR_real``: the difference in medians,
    normalized by the interquartile range for robustness against outliers.
    When the real IQR is 0 (zero-inflated columns) the real standard deviation
    is used instead.

    The median is less sensitive to extreme values than the mean,
    making this metric robust for skewed distributions.
    """

    name: str = "delta_median"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """IQR-normalized median difference in [0, inf), lower is better."""
        q25, q75 = np.percentile(real_col, [25, 75])
        return relative_difference(
            np.median(real_col) - np.median(synth_col), q75 - q25, real_col.std()
        )
