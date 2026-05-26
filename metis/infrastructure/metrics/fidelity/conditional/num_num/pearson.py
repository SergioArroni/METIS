"""
Pearson Correlation metric for Num↔Num pairs.

Measures linear correlation preservation between numeric column pairs.
"""

import warnings

import pandas as pd
from scipy import stats

from ....registry import register
from ...fidelity_base import NumNumPairMetric


@register("fidelity.pearson")
class PearsonCorrelationMetric(NumNumPairMetric):
    """
    Pearson correlation preservation metric.

    Measures how well synthetic data preserves linear correlations
    between pairs of numeric columns.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        >>> synth = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 5, 7, 9, 11]})
        >>> metric = PearsonCorrelationMetric()
        >>> metric.fit(real, synth)
        >>> result = metric.compute()
    """

    name: str = "pearson"

    def _compute_pair(
        self,
        real_col1: pd.Series,
        real_col2: pd.Series,
        synth_col1: pd.Series,
        synth_col2: pd.Series,
    ) -> tuple[float, float]:
        """Compute Pearson correlation for a column pair."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real_val = abs(stats.pearsonr(real_col1.values, real_col2.values)[0])
            synth_val = abs(stats.pearsonr(synth_col1.values, synth_col2.values)[0])

        return real_val, synth_val
