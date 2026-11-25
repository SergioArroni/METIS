"""Delta Mean metric for scale distribution comparison."""

import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric, relative_difference


@register("fidelity.delta_mean")
class DeltaMeanMetric(MarginalMetric):
    """
    Delta Mean - compare means between distributions.

    ``|mean_real - mean_synth| / std_real``: the difference in means in
    standard deviations of the real column (scale-free). A constant real
    column scores 1 when the means agree and 0 otherwise.
    """

    name: str = "delta_mean"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """Standardized mean difference in [0, inf), lower is better."""
        return relative_difference(real_col.mean() - synth_col.mean(), real_col.std())
