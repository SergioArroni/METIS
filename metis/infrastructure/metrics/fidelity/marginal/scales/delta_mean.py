"""Delta Mean metric for scale distribution comparison."""

import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric


@register("fidelity.delta_mean")
class DeltaMeanMetric(MarginalMetric):
    """
    Delta Mean - compare means between distributions.

    Measures the standardized difference in means between real and
    synthetic data. Normalization by standard deviation provides
    scale invariance.

    Similar to a one-sample effect size measure.
    """

    name: str = "delta_mean"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Delta Mean for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Standardized mean difference in [0, ∞), lower is better
        """
        real_mean = real_col.mean()
        synth_mean = synth_col.mean()

        # Normalize by real std for scale invariance
        real_std = real_col.std()
        if real_std > 1e-10:
            return abs(real_mean - synth_mean) / real_std
        return abs(real_mean - synth_mean)
