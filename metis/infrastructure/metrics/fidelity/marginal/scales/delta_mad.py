"""Delta Median Absolute Deviation metric for scale distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric


@register("fidelity.delta_mad")
class DeltaMADMetric(MarginalMetric):
    """
    Delta Median Absolute Deviation - compare MAD between distributions.

    Measures the relative difference in MAD (Median Absolute Deviation)
    between real and synthetic data. The MAD is a robust measure of
    statistical dispersion.

    Result is normalized by real MAD for scale invariance.
    Returns absolute difference if real MAD is near zero.
    """

    name: str = "delta_mad"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Delta MAD for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Relative MAD difference in [0, ∞), lower is better
        """
        real_mad = np.median(np.abs(real_col - np.median(real_col)))
        synth_mad = np.median(np.abs(synth_col - np.median(synth_col)))

        # Normalize by real MAD to get relative difference
        if real_mad > 1e-10:
            return abs(real_mad - synth_mad) / real_mad
        return abs(real_mad - synth_mad)
