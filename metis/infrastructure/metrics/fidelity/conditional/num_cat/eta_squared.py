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
        Compute eta-squared (effect size) from grouped data.

        Eta-squared represents the proportion of total variance that is
        explained by group membership (SS_between / SS_total).

        Args:
            groups: list of numpy arrays, one per group

        Returns:
            Eta-squared value in [0, 1]
        """
        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        return float(ss_between / ss_total) if ss_total > 0 else 0.0

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
        real_df = pd.DataFrame({"num": real_col1, "cat": real_col2})
        synth_df = pd.DataFrame({"num": synth_col1, "cat": synth_col2})

        real_groups = [g["num"].values for _, g in real_df.groupby("cat") if len(g) > 1]
        synth_groups = [g["num"].values for _, g in synth_df.groupby("cat") if len(g) > 1]

        if len(real_groups) < 2 or len(synth_groups) < 2:
            raise ValueError("Need at least 2 groups with >1 element each")

        r_eta = self._compute_eta_squared(real_groups)
        s_eta = self._compute_eta_squared(synth_groups)

        return r_eta, s_eta
