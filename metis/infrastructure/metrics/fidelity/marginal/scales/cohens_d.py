"""Cohen's d effect size metric for scale distribution comparison."""

import numpy as np
import pandas as pd

from metis.infrastructure.metrics.registry import register

from ...fidelity_base import NumericColumnMetric as MarginalMetric


@register("fidelity.cohens_d")
class CohensD(MarginalMetric):
    """
    Cohen's d - standardized effect size between distributions.

    Measures the standardized difference between two means using
    the pooled standard deviation. This is a widely used effect
    size measure in statistics.

    Interpretation guidelines:
    - d = 0.2: small effect
    - d = 0.5: medium effect
    - d = 0.8: large effect

    Values closer to 0 indicate better synthetic data quality.
    """

    name: str = "cohens_d"
    is_distance: bool = True

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute Cohen's d for a single column.

        Args:
            real_col: Column from real dataset
            synth_col: Column from synthetic dataset

        Returns:
            Absolute Cohen's d in [0, ∞), lower is better
        """
        n1, n2 = len(real_col), len(synth_col)
        var1, var2 = real_col.var(), synth_col.var()

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std > 1e-10:
            d = abs(real_col.mean() - synth_col.mean()) / pooled_std
        else:
            d = 0.0

        return d
