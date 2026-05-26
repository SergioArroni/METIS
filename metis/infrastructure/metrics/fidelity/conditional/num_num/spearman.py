"""
Spearman Correlation metric for Num↔Num pairs.

Measures rank-based correlation preservation between numeric column pairs.
"""

import warnings

import pandas as pd
from scipy import stats

from ....registry import register
from ...fidelity_base import NumNumPairMetric


@register("fidelity.spearman")
class SpearmanCorrelationMetric(NumNumPairMetric):
    """
    Spearman correlation preservation metric.

    Measures how well synthetic data preserves rank-based correlations
    between pairs of numeric columns. More robust to outliers and
    captures monotonic (not just linear) relationships.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        >>> synth = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 5, 7, 9, 11]})
        >>> metric = SpearmanCorrelationMetric()
        >>> metric.fit(real, synth)
        >>> result = metric.compute()
    """

    name: str = "spearman"

    def _compute_pair(
        self,
        real_col1: pd.Series,
        real_col2: pd.Series,
        synth_col1: pd.Series,
        synth_col2: pd.Series,
    ) -> tuple[float, float]:
        """Compute Spearman correlation for a column pair."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real_val = abs(stats.spearmanr(real_col1.values, real_col2.values)[0])
            synth_val = abs(stats.spearmanr(synth_col1.values, synth_col2.values)[0])

        return real_val, synth_val
