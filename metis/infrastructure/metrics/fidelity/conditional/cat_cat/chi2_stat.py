"""
Chi-squared based metric for Cat↔Cat pairs.

Compares the bias-corrected squared Cramér's V (``phi^2`` divided by its
theoretical maximum) between real and synthetic column pairs.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ....registry import register
from ...fidelity_base import CatCatPairMetric


def bias_corrected_v2(contingency: np.ndarray) -> float:
    """Bias-corrected squared Cramér's V of a contingency table, in ``[0, 1]``.

    ``phi^2 = chi^2 / n`` (Pearson's chi-squared, without Yates' correction) is
    bounded by ``min(r, c) - 1`` (Cramér 1946), so ``V^2 = phi^2 / (min(r, c) - 1)``
    is in ``[0, 1]`` and does not depend on the number of categories. ``phi^2`` is
    also the total inertia of the correspondence analysis of the table, i.e. the
    sum of its ``min(r, c) - 1`` squared canonical correlations (Greenacre 1984),
    so ``V^2`` is their mean: the categorical analogue of ``r^2``.

    The plug-in estimate is biased upwards by about ``(r - 1)(c - 1) / (n - 1)``,
    which depends on ``n`` and on the cardinality. Bergsma (2013) corrects it::

        phi2~ = max(0, phi^2 - (r - 1)(c - 1) / (n - 1))
        r~ = r - (r - 1)^2 / (n - 1),   c~ = c - (c - 1)^2 / (n - 1)
        V~^2 = phi2~ / min(r~ - 1, c~ - 1)

    Rows/columns with no observations are dropped first. A table with a single
    row or column has no association (0). Returns NaN when the corrected
    dimensions are not positive (fewer observations than categories).

    References:
        Cramér, H. (1946). Mathematical Methods of Statistics. Princeton UP.
        Greenacre, M. J. (1984). Theory and Applications of Correspondence
        Analysis. Academic Press.
        Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
        Journal of the Korean Statistical Society, 42(3), 323-328.
    """
    table = np.asarray(contingency, dtype=float)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    n = table.sum()
    r, c = table.shape if table.ndim == 2 else (0, 0)
    if n <= 1 or min(r, c) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    dim = min(r_corr - 1, c_corr - 1)
    if dim <= 0:
        return float("nan")
    return float(min(phi2_corr / dim, 1.0))


@register("fidelity.chi2_stat")
class Chi2StatMetric(CatCatPairMetric):
    """
    Chi-squared association preservation metric (bias-corrected ``V^2``).

    For each pair of categorical columns the raw value is the bias-corrected
    squared Cramér's V (:func:`bias_corrected_v2`): the chi-squared statistic
    scaled by ``n`` and by its theoretical maximum ``min(r, c) - 1``. The score of
    a pair is ``1 - |V~^2_real - V~^2_synth|`` (a bounded distance: no scale
    parameter). The previous version compared raw ``chi^2 / n`` with
    ``exp(-delta / 0.3)``; ``chi^2 / n`` grows with the number of categories,
    so the same loss of association scored very differently in a 2x2 and in a
    6x8 table.

    The metric is a monotone transform of Cramér's V of the same table; it is
    kept (rather than removed) because it weights errors differently
    (``|dV^2| = |dV| (V_r + V_s)``: errors on strong associations count more)
    and because it is bias-corrected, while ``fidelity.cramers_v`` is not.

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
        Bias-corrected ``V^2`` of a column pair in each table.

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
        return bias_corrected_v2(real_cont), bias_corrected_v2(synth_cont)
