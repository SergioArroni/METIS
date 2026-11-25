"""Hellinger distance metric for tail distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as TailMetric
from ..coverage._binning import quantile_bin_distributions


@register("fidelity.hellinger")
class HellingerMetric(TailMetric):
    """
    Hellinger distance between distributions.

    Measures the similarity between two probability distributions.
    The Hellinger distance is bounded in [0, 1] where:
        - 0 = identical distributions
        - 1 = completely disjoint distributions

    Both columns are binned on the ``n_bins``-quantiles of the real column
    (open-ended outer bins). It used 50 equal-width bins over the pooled data:
    the edges depended on the synthetic table (one synthetic outlier put both
    tables in the same bin and the distance went to ~0) and the plug-in bias
    of 50 bins with a few hundred rows dominated the value.
    """

    name: str = "hellinger"
    is_distance: bool = True  # Lower distance = better

    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = int(n_bins)

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """Hellinger distance in [0, 1] for a single column."""
        if real_col.nunique() == 1 and synth_col.nunique() == 1:
            return 0.0 if real_col.iloc[0] == synth_col.iloc[0] else 1.0
        p, q = quantile_bin_distributions(real_col, synth_col, self.n_bins)
        # Hellinger distance: sqrt(1 - BC) where BC is the Bhattacharyya coefficient
        bc = np.sum(np.sqrt(p * q))
        return float(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0)))
