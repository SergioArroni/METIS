"""Total Variation Distance metric for coverage distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register
from metis.shared.distributions import align_distributions

from ...fidelity_base import UniversalColumnMetric as UniversalMarginalMetric


@register("fidelity.tvd")
class TVDMetric(UniversalMarginalMetric):
    """
    Total Variation Distance between distributions.

    Measures the maximum difference between two probability distributions.
    TVD = 0.5 * sum(|p - q|)

    TVD is bounded in [0, 1]:
    - 0: identical distributions
    - 1: completely disjoint distributions

    Works for both categorical and numeric columns.
    """

    name: str = "tvd"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute TVD for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            TVD in [0, 1], lower is better
        """
        p, q = align_distributions(real_col, synth_col)
        # TVD = 0.5 * sum(|p - q|)
        return 0.5 * np.sum(np.abs(p - q))
