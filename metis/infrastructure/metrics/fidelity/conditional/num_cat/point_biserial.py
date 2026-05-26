"""Point-Biserial Correlation metric for Num<->Cat pairs.

Measures correlation between numeric and binary categorical columns.
"""

import pandas as pd
from scipy.stats import pointbiserialr

from ....registry import register
from ...fidelity_base import NumCatPairMetric


@register("fidelity.point_biserial")
class PointBiserialMetric(NumCatPairMetric):
    """
    Point-Biserial correlation preservation metric.

    Measures how well synthetic data preserves the relationship
    between numeric columns and binary categorical columns.

    Note: Only applicable for binary (2-category) categorical columns.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"score": [80, 90, 70, 85], "passed": ["Y", "Y", "N", "Y"]})
        >>> synth = pd.DataFrame({"score": [82, 88, 72, 84], "passed": ["Y", "Y", "N", "Y"]})
        >>> metric = PointBiserialMetric()
        >>> results = metric.compute(real, synth)
    """

    name: str = "point_biserial"

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
        Compute Point-Biserial correlation for a numeric-categorical pair.

        Args:
            real_col1: Numeric column from real dataset
            real_col2: Categorical column from real dataset (must be binary)
            synth_col1: Numeric column from synthetic dataset
            synth_col2: Categorical column from synthetic dataset (must be binary)

        Returns:
            tuple of (real_point_biserial, synth_point_biserial) - absolute values
        """
        # Only for binary categories
        real_cats = real_col2.unique()
        synth_cats = synth_col2.unique()

        if len(real_cats) != 2 or len(synth_cats) != 2:
            raise ValueError("Point-biserial requires binary categorical columns")

        real_binary = (real_col2 == real_cats[0]).astype(int)
        synth_binary = (synth_col2 == synth_cats[0]).astype(int)

        r_pb, _ = pointbiserialr(real_binary, real_col1)
        s_pb, _ = pointbiserialr(synth_binary, synth_col1)

        return abs(r_pb), abs(s_pb)
