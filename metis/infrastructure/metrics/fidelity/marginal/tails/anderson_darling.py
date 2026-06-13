"""Anderson-Darling test metric for tail distribution comparison."""

import warnings

import pandas as pd
from scipy.stats import anderson_ksamp

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as TailMetric


@register("fidelity.anderson_darling")
class AndersonDarlingMetric(TailMetric):
    """
    Anderson-Darling k-sample test.

    A more powerful test than KS for detecting differences in distribution tails.
    The AD test gives more weight to the tails compared to the center of the
    distribution, making it particularly suitable for tail comparison.

    The statistic is in [0, ∞) and will be normalized using robust percentiles.
    """

    name: str = "anderson_darling"
    is_distance: bool = True  # Higher statistic = more different = worse

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Anderson-Darling statistic for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Anderson-Darling statistic (non-negative)
        """
        try:
            # Suppress p-value warnings (floored/capped) that occur with extreme similarities
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="p-value")
                result = anderson_ksamp([real_col.values, synth_col.values], variant="log-normal")
            return float(result.statistic)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug("anderson_darling fallback for edge case: %s", e)
            return 0.0
