"""Delta Median metric for scale distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric


@register("fidelity.delta_median")
class DeltaMedianMetric(MarginalMetric):
    """
    Delta Median - compare medians between distributions.

    Measures the difference in medians between real and synthetic data,
    normalized by the interquartile range for robustness against outliers.

    The median is less sensitive to extreme values than the mean,
    making this metric robust for skewed distributions.
    """

    name: str = "delta_median"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Delta Median for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            IQR-normalized median difference in [0, ∞), lower is better
        """
        real_median = np.median(real_col)
        synth_median = np.median(synth_col)

        # Normalize by IQR for robustness
        real_iqr = np.percentile(real_col, 75) - np.percentile(real_col, 25)
        if real_iqr > 1e-10:
            return abs(real_median - synth_median) / real_iqr
        return abs(real_median - synth_median)
