"""
Chi-squared statistic metric for Cat↔Cat pairs.

Measures normalized chi-squared statistic between categorical column pairs.
"""

import pandas as pd
from scipy.stats import chi2_contingency

from ....registry import register
from ...fidelity_base import CatCatPairMetric


@register("fidelity.chi2_stat")
class Chi2StatMetric(CatCatPairMetric):
    """
    Normalized Chi-squared statistic preservation metric.

    Measures how well synthetic data preserves the chi-squared
    independence test statistic between pairs of categorical columns.
    The statistic is normalized by sample size.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"gender": ["M", "F", "M", "F"], "status": ["A", "B", "A", "B"]})
        >>> synth = pd.DataFrame({"gender": ["M", "F", "F", "M"], "status": ["A", "B", "B", "A"]})
        >>> metric = Chi2StatMetric()
        >>> result = metric.fit(real, synth).compute()
    """

    name: str = "chi2_stat"

    def __init__(self):
        super().__init__()

    def _compute_pair(
        self,
        real_col1: pd.Series,
        real_col2: pd.Series,
        synth_col1: pd.Series,
        synth_col2: pd.Series,
    ) -> tuple[float, float]:
        """
        Compute normalized Chi-squared for a column pair.

        Args:
            real_col1: First column from real dataset
            real_col2: Second column from real dataset
            synth_col1: First column from synthetic dataset
            synth_col2: Second column from synthetic dataset

        Returns:
            tuple of (real_value, synth_value)
        """
        real_cont = pd.crosstab(real_col1, real_col2).values
        synth_cont = pd.crosstab(synth_col1, synth_col2).values

        r_chi2 = chi2_contingency(real_cont)[0] / len(real_col1)
        s_chi2 = chi2_contingency(synth_cont)[0] / len(synth_col1)

        return r_chi2, s_chi2
