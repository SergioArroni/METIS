"""
Mutual Information (MI) metric for Num↔Num pairs.

Measures information-theoretic dependency preservation between numeric column pairs.
"""

import warnings

import numpy as np
import pandas as pd

from ....registry import register
from ...fidelity_base import NumNumPairMetric


@register("fidelity.mi")
class MutualInformationMetric(NumNumPairMetric):
    """
    Mutual Information preservation metric.

    Measures how well synthetic data preserves information-theoretic
    dependencies between pairs of numeric columns. MI captures any
    kind of statistical dependency.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 4, 9, 16, 25]})
        >>> synth = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 5, 10, 17, 26]})
        >>> metric = MutualInformationMetric()
        >>> metric.fit(real, synth)
        >>> result = metric.compute()
    """

    name: str = "mi"
    bins: int = 10

    def __init__(self, bins: int = 10):
        """
        Initialize MI metric.

        Args:
            bins: Number of bins for discretization
        """
        super().__init__()
        self.bins = bins

    def _compute_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information between two numeric arrays (discretized).

        MI measures the amount of information one variable contains about another.
        Variables are discretized into bins for computation.

        Args:
            x: First numeric array
            y: Second numeric array

        Returns:
            Mutual information value (non-negative)
        """
        # Discretize continuous variables
        x_binned = pd.cut(x, bins=self.bins, labels=False, duplicates="drop")
        y_binned = pd.cut(y, bins=self.bins, labels=False, duplicates="drop")

        # Handle NaN from binning
        mask = ~(np.isnan(x_binned) | np.isnan(y_binned))
        if mask.sum() < 10:
            return 0.0

        x_binned = x_binned[mask].astype(int)
        y_binned = y_binned[mask].astype(int)

        # Compute joint and marginal distributions
        contingency = np.histogram2d(x_binned, y_binned, bins=self.bins)[0]
        pxy = contingency / contingency.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # MI = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = 0.0
            for i in range(len(px)):
                for j in range(len(py)):
                    if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                        mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

        return max(0.0, float(mi))

    def _compute_pair(
        self,
        real_col1: pd.Series,
        real_col2: pd.Series,
        synth_col1: pd.Series,
        synth_col2: pd.Series,
    ) -> tuple[float, float]:
        """Compute Mutual Information for a column pair."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real_val = self._compute_mi(real_col1.values, real_col2.values)
            synth_val = self._compute_mi(synth_col1.values, synth_col2.values)

        return real_val, synth_val
