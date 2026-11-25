"""Total Variation Distance metric for coverage distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register
from metis.shared.distributions import align_distributions

from ...fidelity_base import UniversalColumnMetric as UniversalMarginalMetric
from ._binning import is_continuous, quantile_bin_distributions


@register("fidelity.tvd")
class TVDMetric(UniversalMarginalMetric):
    """
    Total Variation Distance between distributions.

    Measures the maximum difference between two probability distributions.
    TVD = 0.5 * sum(|p - q|)

    TVD is bounded in [0, 1]:
    - 0: identical distributions
    - 1: completely disjoint distributions

    Categorical and discrete columns are compared on their categories.
    Continuous columns (numeric, >= 20 distinct values) are binned on the
    ``n_bins``-quantiles of the real column (open-ended outer bins), the same
    grid as PSI and KL; see ``quantile_bin_distributions``.
    """

    name: str = "tvd"
    is_distance: bool = True

    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = int(n_bins)

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """TVD in [0, 1] for a single column (lower is better)."""
        if is_continuous(real_col):
            p, q = quantile_bin_distributions(real_col, synth_col, self.n_bins)
        else:
            p, q = align_distributions(real_col, synth_col)
        return float(0.5 * np.sum(np.abs(p - q)))
