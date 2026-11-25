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

    @staticmethod
    def _edges(values: np.ndarray, bins: int) -> np.ndarray:
        """Interior bin edges: quantiles of the *real* column (equal-frequency)."""
        return np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))

    @staticmethod
    def _mi_miller_madow(x_codes: np.ndarray, y_codes: np.ndarray, kx: int, ky: int) -> float:
        """Plug-in MI with the Miller-Madow bias correction.

        Each entropy gets ``+ (m - 1) / (2n)`` with ``m`` the number of non-empty
        cells, so ``MI_MM = MI_plugin - (m_xy - m_x - m_y + 1) / (2n)``. The
        plug-in estimator alone is biased upwards by ~0.1 nats with n=400 and
        10x10 bins, even for independent columns. Clipped at 0.
        """
        n = len(x_codes)
        joint = np.bincount(x_codes * ky + y_codes, minlength=kx * ky).reshape(kx, ky)
        pxy = joint / n
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        nz = pxy > 0
        outer = np.outer(px, py)
        mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / outer[nz])))
        m_xy = int(nz.sum())
        m_x = int((px > 0).sum())
        m_y = int((py > 0).sum())
        mi -= (m_xy - m_x - m_y + 1) / (2.0 * n)
        return max(0.0, mi)

    def _compute_mi(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_edges: np.ndarray | None = None,
        y_edges: np.ndarray | None = None,
    ) -> float:
        """
        Bias-corrected mutual information between two numeric arrays.

        Both arrays are discretised on fixed edges; when none are given they
        are fitted on the arrays themselves. In ``_compute_pair`` the edges are
        always fitted on the real data and reused for the synthetic data, so
        both tables are measured on the same grid (values outside fall in the
        open-ended outer bins).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 10:
            return 0.0
        x, y = x[mask], y[mask]
        if x_edges is None:
            x_edges = self._edges(x, self.bins)
        if y_edges is None:
            y_edges = self._edges(y, self.bins)
        x_codes = np.searchsorted(x_edges, x, side="right")
        y_codes = np.searchsorted(y_edges, y, side="right")
        return self._mi_miller_madow(x_codes, y_codes, len(x_edges) + 1, len(y_edges) + 1)

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
            x_edges = self._edges(real_col1.to_numpy(dtype=float), self.bins)
            y_edges = self._edges(real_col2.to_numpy(dtype=float), self.bins)
            real_val = self._compute_mi(real_col1.values, real_col2.values, x_edges, y_edges)
            synth_val = self._compute_mi(synth_col1.values, synth_col2.values, x_edges, y_edges)

        return real_val, synth_val
