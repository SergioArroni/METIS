"""Kolmogorov-Smirnov test metric for tail distribution comparison."""

import pandas as pd
from scipy.stats import ks_2samp

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as TailMetric


@register("fidelity.ks")
class KSMetric(TailMetric):
    """
    Kolmogorov-Smirnov two-sample test.

    Measures the maximum difference between the empirical CDFs of real
    and synthetic data. The statistic is in [0, 1] where 0 = identical
    distributions and 1 = completely different.

    This metric is particularly sensitive to differences in the tails
    and is distribution-free (non-parametric).
    """

    name: str = "ks"
    is_distance: bool = True  # Lower KS statistic = better

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute KS statistic for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            KS statistic in [0, 1]
        """
        statistic, _ = ks_2samp(real_col.values, synth_col.values)
        return float(statistic)
