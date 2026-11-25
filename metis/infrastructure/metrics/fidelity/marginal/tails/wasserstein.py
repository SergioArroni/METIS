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

    The raw distance is in the units of the column; it is divided by the real
    column's standard deviation before the absolute normalization
    ``exp(-(W/σ_real) / scale)`` of the central registry.
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

    def _scale_free_value(self, raw_value: float, real_col: pd.Series) -> float:
        """Express the distance in standard deviations of the real column."""
        sigma = float(real_col.std())
        if not sigma > 1e-12:
            # Constant real column: any displacement is total, none is perfect.
            return 0.0 if raw_value <= 1e-12 else float("inf")
        return raw_value / sigma
