"""Jensen-Shannon Divergence metric for coverage distribution comparison."""

import numpy as np
import pandas as pd
from scipy.special import rel_entr

from metis.infrastructure.metrics.registry import register
from metis.shared.distributions import align_distributions

from ...fidelity_base import UniversalColumnMetric as UniversalMarginalMetric


@register("fidelity.js")
class JSDivergenceMetric(UniversalMarginalMetric):
    """
    Jensen-Shannon Divergence (symmetric, bounded).

    A symmetrized and smoothed version of KL divergence.
    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P+Q)/2

    Returns the JS distance (square root of JS divergence).
    Bounded in [0, 1] for base-2 logarithm.

    Works for both categorical and numeric columns.
    """

    name: str = "js"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute JS Divergence for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            JS distance in [0, 1], lower is better
        """
        p, q = align_distributions(real_col, synth_col)
        # JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P+Q)/2
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
        return np.sqrt(js)  # Return JS distance (sqrt of divergence)
