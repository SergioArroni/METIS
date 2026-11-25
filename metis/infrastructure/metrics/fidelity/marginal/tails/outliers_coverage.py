"""
Outliers and Structural Coverage metrics.

Measures how well synthetic data covers the structural aspects of real data:
- Outlier coverage: Do synthetic outliers match real outliers?
- Range coverage: Does synthetic data cover the same value ranges?
- Density coverage: Does synthetic data match real data density regions?
"""

import numpy as np
import pandas as pd

from ....registry import register
from ...fidelity_base import GlobalFidelityMetric, GlobalMetricNotApplicable


@register("fidelity.outliers_coverage")
class OutliersCoverageMetric(GlobalFidelityMetric):
    """
    Measures outlier and structural coverage between real and synthetic data.

    This metric evaluates:
    1. Range coverage: Proportion of real data range covered by synthetic
    2. Outlier overlap: How well synthetic outliers match real outliers
    3. Density overlap: Coverage of high-density regions

    The final score is a weighted combination of these aspects.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> real = pd.DataFrame({"a": np.random.randn(1000)})
        >>> synth = pd.DataFrame({"a": np.random.randn(1000)})
        >>> metric = OutliersCoverageMetric()
        >>> metric.fit(real, synth)
        >>> result = metric.compute()
        >>> print(f"Coverage Score: {result.value:.4f}")
    """

    name: str = "outliers_coverage"

    def __init__(
        self,
        outlier_method: str = "iqr",
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.0,
    ):
        """
        Initialize the metric.

        Args:
            outlier_method: Method for detecting outliers ("iqr" or "zscore")
            iqr_factor: Factor for IQR method (default 1.5)
            zscore_threshold: Threshold for z-score method (default 3.0)
        """
        super().__init__()
        self.outlier_method = outlier_method
        self.iqr_factor = iqr_factor
        self.zscore_threshold = zscore_threshold

    def _compute_global(self) -> tuple[float, float, dict]:
        """
        Compute outliers and coverage metric.

        Returns:
            tuple of (raw_value, normalized_value, details)
        """
        # Common numeric columns (sorted); an ordinal enters by its rank
        # ``__ord`` (its categorical twin is not numeric, so it counts once).
        common_cols = self._get_numeric_columns(include_ordinal_ranks=True)
        real_num, synth_num = self._real_data, self._synth_data

        if not common_cols:
            raise GlobalMetricNotApplicable("No common numeric columns")

        # Compute sub-metrics
        range_coverage = self._compute_range_coverage(real_num[common_cols], synth_num[common_cols])
        outlier_overlap = self._compute_outlier_overlap(
            real_num[common_cols], synth_num[common_cols]
        )
        density_coverage = self._compute_density_coverage(
            real_num[common_cols], synth_num[common_cols]
        )

        # Weighted combination
        score = 0.3 * range_coverage + 0.4 * outlier_overlap + 0.3 * density_coverage

        details = {
            "range_coverage": range_coverage,
            "outlier_overlap": outlier_overlap,
            "density_coverage": density_coverage,
            "n_columns": len(common_cols),
            "outlier_method": self.outlier_method,
        }

        return score, score, details  # Already in [0, 1]

    # Quantiles of the real column that delimit its "range" (robust to the
    # single most extreme draws, which are pure sampling noise).
    range_quantiles: tuple[float, float] = (0.01, 0.99)
    density_bins: int = 10

    def _compute_range_coverage(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """Share of the real robust range ``[q01, q99]`` covered by the synthetic one."""
        lo_q, hi_q = self.range_quantiles
        coverages = []
        for col in real_df.columns:
            real = real_df[col].dropna()
            synth = synth_df[col].dropna()
            if real.empty or synth.empty:
                continue
            real_min, real_max = real.quantile(lo_q), real.quantile(hi_q)
            synth_min, synth_max = synth.quantile(lo_q), synth.quantile(hi_q)

            if real_max - real_min < 1e-10:
                coverages.append(1.0)
                continue

            overlap = max(0.0, min(real_max, synth_max) - max(real_min, synth_min))
            coverages.append(min(overlap / (real_max - real_min), 1.0))

        return float(np.mean(coverages)) if coverages else float("nan")

    def _compute_outlier_overlap(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """Agreement of the tail masses beyond the fences fitted on the real column.

        Both tables are judged with the *same* fences (from the real data); each
        table used to get its own, so a shifted synthetic column still looked
        "as outlying as the real one". Per column the lower and upper tail
        masses are compared with a Bray-Curtis similarity on counts smoothed by
        +0.5 (``1 - sum|r - s| / sum(r + s)``); no outliers in either table is
        perfect agreement.
        """
        overlaps = []
        for col in real_df.columns:
            real = real_df[col].dropna().to_numpy(dtype=float)
            synth = synth_df[col].dropna().to_numpy(dtype=float)
            if len(real) == 0 or len(synth) == 0:
                continue
            lower, upper = self._fences(real)
            r = np.array([(real < lower).sum(), (real > upper).sum()], dtype=float)
            s = np.array([(synth < lower).sum(), (synth > upper).sum()], dtype=float)
            if r.sum() == 0 and s.sum() == 0:
                overlaps.append(1.0)
                continue
            r_rate = (r + 0.5) / (len(real) + 1.0)
            s_rate = (s + 0.5) / (len(synth) + 1.0)
            overlaps.append(float(1.0 - np.abs(r_rate - s_rate).sum() / (r_rate + s_rate).sum()))

        return float(np.mean(overlaps)) if overlaps else float("nan")

    def _compute_density_coverage(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """Histogram overlap on bins fitted on the real column (its deciles).

        Equal-frequency bins of the real data are shared by both tables (outer
        bins open-ended). 50 equal-width bins over the pooled data left many
        near-empty bins with a few hundred rows, so data from the same process
        only reached ~0.8.
        """
        coverages = []
        quantiles = np.linspace(0, 1, self.density_bins + 1)[1:-1]
        for col in real_df.columns:
            real = real_df[col].dropna().to_numpy(dtype=float)
            synth = synth_df[col].dropna().to_numpy(dtype=float)
            if len(real) == 0 or len(synth) == 0:
                continue
            edges = np.unique(np.quantile(real, quantiles))
            p = np.bincount(np.searchsorted(edges, real, side="right"), minlength=len(edges) + 1)
            q = np.bincount(np.searchsorted(edges, synth, side="right"), minlength=len(edges) + 1)
            coverages.append(float(np.minimum(p / p.sum(), q / q.sum()).sum()))

        return float(np.mean(coverages)) if coverages else float("nan")

    def _fences(self, real: np.ndarray) -> tuple[float, float]:
        """Outlier fences fitted on the real column."""
        if self.outlier_method == "iqr":
            q1, q3 = np.quantile(real, [0.25, 0.75])
            iqr = q3 - q1
            return q1 - self.iqr_factor * iqr, q3 + self.iqr_factor * iqr
        mean, std = float(np.mean(real)), float(np.std(real))
        return mean - self.zscore_threshold * std, mean + self.zscore_threshold * std
