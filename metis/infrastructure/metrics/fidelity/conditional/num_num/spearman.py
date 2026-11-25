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
        corr = getattr(self, "_corr_matrices", None)
        if corr is not None:
            # ``DataFrame.corr`` uses pairwise-complete rows, exactly the rows
            # the pair loop hands over, so the lookup equals the per-pair call.
            real_corr, synth_corr = corr
            c1, c2 = real_col1.name, real_col2.name
            return float(real_corr.at[c1, c2]), float(synth_corr.at[c1, c2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real_val = float(stats.spearmanr(real_col1.values, real_col2.values)[0])
            synth_val = float(stats.spearmanr(synth_col1.values, synth_col2.values)[0])

        return real_val, synth_val

    def _prepare_pairs(self) -> None:
        """One ``DataFrame.corr`` per table instead of one scipy call per pair."""
        cols = sorted({c for pair in self._pairs for c in pair})
        self._corr_matrices = None
        if cols:
            self._corr_matrices = (
                self._real_data[cols].corr(method="spearman"),
                self._synth_data[cols].corr(method="spearman"),
            )
