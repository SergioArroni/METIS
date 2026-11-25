"""Jensen-Shannon Divergence metric for coverage distribution comparison."""

import numpy as np
import pandas as pd
from scipy.special import rel_entr

from metis.infrastructure.metrics.registry import register
from metis.shared.distributions import align_distributions

from ...fidelity_base import UniversalColumnMetric as UniversalMarginalMetric
from ._binning import is_continuous, quantile_bin_distributions


@register("fidelity.js")
class JSDivergenceMetric(UniversalMarginalMetric):
    """
    Jensen-Shannon Divergence (symmetric, bounded).

    A symmetrized and smoothed version of KL divergence.
    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P+Q)/2

    Returns the JS distance (square root of JS divergence).

    Categorical and discrete columns are compared on their categories.
    Continuous columns (numeric, >= 20 distinct values) are binned on the
    ``n_bins``-quantiles of the real column (open-ended outer bins), the same
    grid as PSI and KL; see ``quantile_bin_distributions``.
    """

    name: str = "js"
    is_distance: bool = True

    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = int(n_bins)

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """JS distance for a single column (lower is better)."""
        if is_continuous(real_col):
            p, q = quantile_bin_distributions(real_col, synth_col, self.n_bins)
        else:
            p, q = align_distributions(real_col, synth_col)
        # JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P+Q)/2
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
        # ``rel_entr`` uses the natural log (max ln 2): divide by ln 2 so the
        # distance is the base-2 one, in [0, 1]. In nats disjoint supports
        # gave sqrt(ln 2) = 0.83 and the score never went below 0.17.
        return float(np.sqrt(max(js, 0.0) / np.log(2.0)))
