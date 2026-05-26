"""Kruskal-Wallis Epsilon-Squared metric for Num<->Cat pairs.

Measures non-parametric effect size of categorical variable on numeric variable.
"""

import pandas as pd
from scipy.stats import kruskal

from ....registry import register
from ...fidelity_base import NumCatPairMetric


@register("fidelity.kruskal_epsilon")
class KruskalEpsilonMetric(NumCatPairMetric):
    """
    Kruskal-Wallis Epsilon-Squared preservation metric.

    Measures how well synthetic data preserves the non-parametric
    effect size of categorical groups on numeric variables.
    Uses Kruskal-Wallis H statistic normalized by sample size.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame(
        ...     {"score": [80, 90, 70, 85, 75, 95], "group": ["A", "A", "B", "B", "C", "C"]}
        ... )
        >>> synth = pd.DataFrame(
        ...     {"score": [82, 88, 72, 84, 74, 96], "group": ["A", "A", "B", "B", "C", "C"]}
        ... )
        >>> metric = KruskalEpsilonMetric()
        >>> results = metric.compute(real, synth)
    """

    name: str = "kruskal_epsilon"

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
        Compute Kruskal-Wallis Epsilon-Squared for a numeric-categorical pair.

        Args:
            real_col1: Numeric column from real dataset
            real_col2: Categorical column from real dataset
            synth_col1: Numeric column from synthetic dataset
            synth_col2: Categorical column from synthetic dataset

        Returns:
            tuple of (real_epsilon_squared, synth_epsilon_squared)
        """
        # Build groups for real data
        real_df = pd.DataFrame({"num": real_col1, "cat": real_col2})
        synth_df = pd.DataFrame({"num": synth_col1, "cat": synth_col2})

        real_groups = [g["num"].values for _, g in real_df.groupby("cat") if len(g) > 1]
        synth_groups = [g["num"].values for _, g in synth_df.groupby("cat") if len(g) > 1]

        if len(real_groups) < 2 or len(synth_groups) < 2:
            raise ValueError("Need at least 2 groups with >1 element each")

        r_stat, _ = kruskal(*real_groups)
        s_stat, _ = kruskal(*synth_groups)

        # Epsilon-squared = H / (n - 1)
        r_eps = r_stat / (len(real_df) - 1)
        s_eps = s_stat / (len(synth_df) - 1)

        return r_eps, s_eps
