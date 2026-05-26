"""Delta IQR metric for scale distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric


@register("fidelity.delta_iqr")
class DeltaIQRMetric(MarginalMetric):
    """
    Delta IQR - compare interquartile ranges between distributions.

    Measures the relative difference in IQR (Interquartile Range)
    between real and synthetic data. The IQR represents the spread
    of the middle 50% of values.

    IQR is a robust measure of scale, resistant to outliers.
    """

    name: str = "delta_iqr"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Delta IQR for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Relative IQR difference in [0, ∞), lower is better
        """
        real_iqr = np.percentile(real_col, 75) - np.percentile(real_col, 25)
        synth_iqr = np.percentile(synth_col, 75) - np.percentile(synth_col, 25)

        if real_iqr > 1e-10:
            return abs(real_iqr - synth_iqr) / real_iqr
        return abs(real_iqr - synth_iqr)
