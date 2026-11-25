"""Shannon Entropy Delta metric for coverage distribution comparison."""

import numpy as np
import pandas as pd
from scipy import stats

from metis.infrastructure.metrics.registry import register
from metis.shared.distributions import align_distributions

from ...fidelity_base import UniversalColumnMetric as UniversalMarginalMetric
from ._binning import is_continuous


@register("fidelity.entropy_delta")
class ShannonEntropyDeltaMetric(UniversalMarginalMetric):
    """
    Delta of Shannon Entropy between distributions.

    Measures the relative difference in Shannon entropy (information content)
    between real and synthetic distributions: ``|H(P) - H(Q)| / H(P)``.

    Shannon entropy: H(P) = -sum(p * log(p))

    High entropy indicates more uniform/diverse distribution.
    Low entropy indicates more concentrated distribution.

    Categorical and discrete columns use the entropy of their categories.
    Continuous columns use a histogram whose bin *width* is fixed by the real
    column (``n_bins`` equal-width bins over its range, extended with bins of
    the same width wherever the synthetic data falls). With a common width the
    histogram entropies differ by the differential entropies, so a column that
    is wider or narrower than the real one is detected, and the grid never
    depends on the synthetic table: it used 50 bins over the pooled range,
    where one synthetic outlier squeezed the real column into a single bin.
    """

    name: str = "entropy_delta"
    is_distance: bool = True

    def __init__(self, n_bins: int = 50):
        super().__init__()
        self.n_bins = int(n_bins)

    @staticmethod
    def _binned_entropy(values: np.ndarray, origin: float, width: float, n_bins: int) -> float:
        z = (values - origin) / width
        codes = np.floor(z)
        codes[z == n_bins] = n_bins - 1  # the real maximum closes the last bin
        _, counts = np.unique(codes, return_counts=True)
        return float(stats.entropy(counts))

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """Relative entropy difference in [0, inf), lower is better."""
        if is_continuous(real_col):
            real = real_col.to_numpy(dtype=float)
            synth = synth_col.to_numpy(dtype=float)
            lo, hi = float(real.min()), float(real.max())
            width = (hi - lo) / self.n_bins
            h_real = self._binned_entropy(real, lo, width, self.n_bins)
            h_synth = self._binned_entropy(synth, lo, width, self.n_bins)
        else:
            p, q = align_distributions(real_col, synth_col)
            h_real = stats.entropy(p)
            h_synth = stats.entropy(q)

        if h_real > 1e-10:
            return abs(h_real - h_synth) / h_real
        return abs(h_real - h_synth)
