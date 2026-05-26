"""Hellinger distance metric for tail distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as TailMetric


@register("fidelity.hellinger")
class HellingerMetric(TailMetric):
    """
    Hellinger distance between distributions.

    Measures the similarity between two probability distributions.
    The Hellinger distance is bounded in [0, 1] where:
        - 0 = identical distributions
        - 1 = completely disjoint distributions

    Computed using histogram-based density estimation.
    """

    name: str = "hellinger"
    is_distance: bool = True  # Lower distance = better
    n_bins: int = 50  # Number of bins for histogram estimation

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Hellinger distance for a single column.

        Uses histogram-based estimation of probability densities.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Hellinger distance in [0, 1]
        """
        # Determine common bin edges
        all_data = np.concatenate([real_col.values, synth_col.values])
        min_val, max_val = np.min(all_data), np.max(all_data)

        if min_val == max_val:
            return 0.0  # Constant columns are identical

        bins = np.linspace(min_val, max_val, self.n_bins + 1)

        # Compute normalized histograms (probability mass)
        hist_real, _ = np.histogram(real_col.values, bins=bins, density=True)
        hist_synth, _ = np.histogram(synth_col.values, bins=bins, density=True)

        # Normalize to sum to 1 (proper PMF)
        bin_width = bins[1] - bins[0]
        p = hist_real * bin_width
        q = hist_synth * bin_width

        # Add small epsilon to avoid numerical issues
        eps = 1e-10
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()

        # Hellinger distance: sqrt(1 - BC) where BC is Bhattacharyya coefficient
        bc = np.sum(np.sqrt(p * q))
        # Clip to avoid numerical issues when bc > 1.0 due to floating point precision
        hellinger = np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))

        return float(np.clip(hellinger, 0.0, 1.0))
