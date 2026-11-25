"""Eta-Squared metric for Num<->Cat pairs.

Measures effect size of categorical variable on numeric variable (from ANOVA).
"""

import numpy as np
import pandas as pd

from ....registry import register
from ...fidelity_base import NumCatPairMetric


@register("fidelity.eta_squared")
class EtaSquaredMetric(NumCatPairMetric):
    """
    Eta-Squared preservation metric.

    Measures how well synthetic data preserves the effect size
    (proportion of variance explained) of categorical groups
    on numeric variables.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame(
        ...     {"score": [80, 90, 70, 85, 75, 95], "group": ["A", "A", "B", "B", "C", "C"]}
        ... )
        >>> synth = pd.DataFrame(
        ...     {"score": [82, 88, 72, 84, 74, 96], "group": ["A", "A", "B", "B", "C", "C"]}
        ... )
        >>> metric = EtaSquaredMetric()
        >>> results = metric.compute(real, synth)
    """

    name: str = "eta_squared"

    def __init__(self):
        super().__init__()

    def _compute_eta_squared(self, groups: list[np.ndarray]) -> float:
        """
        Compute the effect size from grouped data, bias-corrected (omega²).

        Eta-squared, ``SS_between / SS_total``, is the proportion of variance
        explained by group membership, but it is biased upwards by about
        ``(k − 1) / (N − 1)``: groups drawn from one population still explain
        some variance by chance, more with fewer rows.  Omega-squared (Hays
        1963) removes that expected share::

            ω² = (SS_between − (k − 1)·MS_within) / (SS_total + MS_within)

        clipped to [0, 1].  It is ≈ 0 for unrelated columns at any N and
        ≈ eta² when N is large.

        Args:
            groups: list of numpy arrays, one per group

        Returns:
            Eta-squared value in [0, 1]
        """
        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        k, n = len(groups), len(all_vals)
        if ss_total <= 0 or n <= k:
            return 0.0
        ms_within = (ss_total - ss_between) / (n - k)
        omega = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
        return float(min(1.0, max(0.0, omega)))

    def _compute_pair(
        self,
        real_col1: pd.Series,
        real_col2: pd.Series,
        synth_col1: pd.Series,
        synth_col2: pd.Series,
    ) -> tuple[float, float]:
        """
        Compute Eta-Squared for a numeric-categorical pair.

        Args:
            real_col1: Numeric column from real dataset
            real_col2: Categorical column from real dataset
            synth_col1: Numeric column from synthetic dataset
            synth_col2: Categorical column from synthetic dataset

        Returns:
            tuple of (real_eta_squared, synth_eta_squared)
        """
        # Build groups for real data

        real_groups = self._groups(real_col1, real_col2)
        synth_groups = self._groups(synth_col1, synth_col2)

        if len(real_groups) < 2 or len(synth_groups) < 2:
            raise ValueError("Need at least 2 groups with >1 element each")

        r_eta = self._compute_eta_squared(real_groups)
        s_eta = self._compute_eta_squared(synth_groups)

        return r_eta, s_eta
