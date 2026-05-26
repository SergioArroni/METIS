"""
Theil's U metric for Cat↔Cat pairs.

Measures asymmetric uncertainty coefficient between categorical column pairs.
"""

import numpy as np
import pandas as pd

from ....registry import register
from ...fidelity_base import CatCatPairMetric


@register("fidelity.theils_u")
class TheilsUMetric(CatCatPairMetric):
    """
    Theil's U preservation metric.

    Measures how well synthetic data preserves the asymmetric
    uncertainty coefficient between pairs of categorical columns.
    U(Y|X) measures how much knowing X reduces uncertainty about Y.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"gender": ["M", "F", "M", "F"], "status": ["A", "B", "A", "B"]})
        >>> synth = pd.DataFrame({"gender": ["M", "F", "F", "M"], "status": ["A", "B", "B", "A"]})
        >>> metric = TheilsUMetric()
        >>> result = metric.fit(real, synth).compute()
    """

    name: str = "theils_u"

    def __init__(self):
        super().__init__()

    def _theils_u(self, contingency: np.ndarray) -> float:
        """
        Compute Theil's U (uncertainty coefficient) from a contingency table.

        Theil's U measures the fraction of uncertainty in Y that is removed
        by knowing X. It is asymmetric: U(Y|X) ≠ U(X|Y).

        Formula: U(Y|X) = (H(Y) - H(Y|X)) / H(Y)

        Args:
            contingency: Contingency table as 2D numpy array

        Returns:
            Theil's U value in [0, 1]
        """
        pxy = contingency / contingency.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # H(Y)
        hy = -np.sum(py[py > 0] * np.log(py[py > 0]))

        # H(Y|X) = sum_x p(x) * H(Y|X=x)
        hy_given_x = 0
        for i in range(len(px)):
            if px[i] > 0:
                py_given_x = pxy[i, :] / px[i]
                py_given_x = py_given_x[py_given_x > 0]
                hy_given_x -= px[i] * np.sum(py_given_x * np.log(py_given_x))

        if hy == 0:
            return 0.0
        return float((hy - hy_given_x) / hy)

    def _compute_pair(
        self,
        real_col1: pd.Series,
        real_col2: pd.Series,
        synth_col1: pd.Series,
        synth_col2: pd.Series,
    ) -> tuple[float, float]:
        """
        Compute Theil's U for a column pair.

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

        r_tu = self._theils_u(real_cont)
        s_tu = self._theils_u(synth_cont)

        return r_tu, s_tu
