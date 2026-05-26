"""Wasserstein (Earth Mover's) distance metric for tail distribution comparison."""

import pandas as pd
from scipy.stats import wasserstein_distance

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as TailMetric


@register("fidelity.wasserstein")
class WassersteinMetric(TailMetric):
    """
    Wasserstein distance (Earth Mover's Distance).

    Measures the minimum "cost" of transforming one distribution into another.
    Unlike KS, it considers the full shape of distributions and is sensitive
    to differences in location, scale, and shape.

    The raw distance is in [0, ∞) and will be normalized using robust percentiles.
    """

    name: str = "wasserstein"
    is_distance: bool = True  # Lower distance = better

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Wasserstein distance for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Wasserstein distance (non-negative)
        """
        distance = wasserstein_distance(real_col.values, synth_col.values)
        return float(distance)
