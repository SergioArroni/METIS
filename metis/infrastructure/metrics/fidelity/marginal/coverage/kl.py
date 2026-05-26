"""Kullback-Leibler Divergence metric for coverage distribution comparison."""

import numpy as np
import pandas as pd
from scipy.special import rel_entr

from metis.infrastructure.metrics.registry import register
from metis.shared.distributions import align_distributions

from ...fidelity_base import UniversalColumnMetric as UniversalMarginalMetric


@register("fidelity.kl")
class KLDivergenceMetric(UniversalMarginalMetric):
    """
    Kullback-Leibler Divergence (asymmetric).

    Measures the relative entropy from synthetic distribution Q to
    real distribution P. KL(P || Q) quantifies the information lost
    when Q is used to approximate P.

    Note: KL divergence is asymmetric - KL(P||Q) ≠ KL(Q||P).
    Values are clipped at 10.0 for numerical stability.

    Works for both categorical and numeric columns.
    """

    name: str = "kl"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute KL Divergence for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            KL divergence in [0, 10], lower is better
        """
        p, q = align_distributions(real_col, synth_col)
        # KL(P || Q) = sum(p * log(p/q))
        kl = np.sum(rel_entr(p, q))
        return min(kl, 10.0)  # Clip extreme values
