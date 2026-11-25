"""
Cramér's V metric for Cat↔Cat pairs.

Measures symmetric association between categorical column pairs.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ....registry import register
from ...fidelity_base import CatCatPairMetric
from .chi2_stat import bias_corrected_v2


@register("fidelity.cramers_v")
class CramersVMetric(CatCatPairMetric):
    """
    Cramér's V preservation metric.

    Measures how well synthetic data preserves the symmetric
    association between pairs of categorical columns.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"gender": ["M", "F", "M", "F"], "status": ["A", "B", "A", "B"]})
        >>> synth = pd.DataFrame({"gender": ["M", "F", "F", "M"], "status": ["A", "B", "B", "A"]})
        >>> metric = CramersVMetric()
        >>> result = metric.fit(real, synth).compute()
    """

    name: str = "cramers_v"

    def __init__(self):
        super().__init__()

    def _cramers_v(self, contingency: np.ndarray) -> float:
        """
        Compute Cramér's V from a contingency table.

        Cramér's V is a measure of association between two categorical variables,
        normalized to lie between 0 and 1.

        Args:
            contingency: Contingency table as 2D numpy array

        Returns:
            Cramér's V value in [0, 1]

        Bias-corrected (Bergsma 2013, ``bias_corrected_v2``): the plug-in V is
        biased upwards by an amount that grows with the cardinality and shrinks
        with n, so tables of different sizes (or with gaps in different
        columns) were compared with different biases.  Falls back to the
        plug-in V when the correction is undefined (fewer rows than levels).
        """
        corrected = bias_corrected_v2(contingency)
        if np.isfinite(corrected):
            return float(np.sqrt(corrected))
        # Uncorrected chi^2 (the definition of V): scipy applies Yates'
        # continuity correction to 2x2 tables by default, which shrinks V by an
        # amount that depends on n.
        chi2 = chi2_contingency(contingency, correction=False)[0]
        n = contingency.sum()
        min_dim = min(contingency.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))

    def _compute_pair(
        self,
        real_col1: pd.Series,
        real_col2: pd.Series,
        synth_col1: pd.Series,
        synth_col2: pd.Series,
    ) -> tuple[float, float]:
        """
        Compute Cramér's V for a column pair.

        Args:
            real_col1: First column from real dataset
            real_col2: Second column from real dataset
            synth_col1: First column from synthetic dataset
            synth_col2: Second column from synthetic dataset

        Returns:
            tuple of (real_value, synth_value)
        """
        real_cont = self._contingency("real", real_col1, real_col2)
        synth_cont = self._contingency("synth", synth_col1, synth_col2)

        r_cv = self._cramers_v(real_cont)
        s_cv = self._cramers_v(synth_cont)

        return r_cv, s_cv
