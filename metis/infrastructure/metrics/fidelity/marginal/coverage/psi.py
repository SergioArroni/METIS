"""Population Stability Index metric for coverage distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register
from metis.shared.distributions import align_distributions

from ...fidelity_base import UniversalColumnMetric as UniversalMarginalMetric
from ._binning import is_continuous, quantile_bin_distributions


@register("fidelity.psi")
class PSIMetric(UniversalMarginalMetric):
    """
    Population Stability Index (PSI).

    Measures the shift in population distribution between real and
    synthetic data. Originally used in credit scoring to detect
    population drift.

    PSI = sum((p - q) * ln(p/q))

    Interpretation guidelines:
    - PSI < 0.1: no significant change
    - 0.1 ≤ PSI < 0.25: moderate change
    - PSI ≥ 0.25: significant change

    Numeric columns are binned on the deciles of the real column (the usual
    PSI construction, open-ended outer bins), not on 50 equal-width bins of the
    pooled data: with a few hundred rows those leave many near-empty bins and
    the plug-in bias alone pushed PSI to ~0.3 for data of the same process.

    Values are clipped at 10.0 for numerical stability.
    Works for both categorical and numeric columns.
    """

    name: str = "psi"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute PSI for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            PSI in [0, 10], lower is better
        """
        if is_continuous(real_col):
            p, q = quantile_bin_distributions(real_col, synth_col)
        else:
            p, q = align_distributions(real_col, synth_col)

        # PSI = sum((p - q) * ln(p/q))
        with np.errstate(divide="ignore", invalid="ignore"):
            psi = np.sum((p - q) * np.log(p / q))

        # Handle NaN and inf
        if np.isnan(psi) or np.isinf(psi):
            psi = 10.0  # Max penalty

        return min(psi, 10.0)
