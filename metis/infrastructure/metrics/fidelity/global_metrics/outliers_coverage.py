"""
Outliers and Structural Coverage metrics.

Measures how well synthetic data covers the structural aspects of real data:
- Outlier coverage: Do synthetic outliers match real outliers?
- Range coverage: Does synthetic data cover the same value ranges?
- Density coverage: Does synthetic data match real data density regions?
"""

import numpy as np
import pandas as pd
from scipy import stats

from ...registry import register
from ..fidelity_base import GlobalFidelityMetric


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
        # Get numeric columns
        real_num = self._real_data.select_dtypes(include=[np.number])
        synth_num = self._synth_data.select_dtypes(include=[np.number])
        common_cols = list(set(real_num.columns) & set(synth_num.columns))

        if not common_cols:
            raise ValueError("No common numeric columns")

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

    def _compute_range_coverage(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """Compute how well synthetic data covers real data range."""
        coverages = []

        for col in real_df.columns:
            real_min, real_max = real_df[col].min(), real_df[col].max()
            synth_min, synth_max = synth_df[col].min(), synth_df[col].max()

            if real_max - real_min < 1e-10:
                coverages.append(1.0)
                continue

            # Coverage = overlap / real range
            overlap_min = max(real_min, synth_min)
            overlap_max = min(real_max, synth_max)
            overlap = max(0, overlap_max - overlap_min)

            coverage = overlap / (real_max - real_min)
            coverages.append(min(coverage, 1.0))

        return float(np.mean(coverages))

    def _compute_outlier_overlap(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """Compute overlap between real and synthetic outliers."""
        overlaps = []

        for col in real_df.columns:
            real_outliers = self._detect_outliers(real_df[col])
            synth_outliers = self._detect_outliers(synth_df[col])

            # Jaccard similarity of outlier positions
            n_real = real_outliers.sum()
            n_synth = synth_outliers.sum()

            if n_real == 0 and n_synth == 0:
                overlaps.append(1.0)  # No outliers in either = perfect match
            elif n_real == 0 or n_synth == 0:
                overlaps.append(0.5)  # One has outliers, other doesn't
            else:
                # Compare outlier value ranges
                real_outlier_vals = real_df[col][real_outliers]
                synth_outlier_vals = synth_df[col][synth_outliers]

                # Check if synthetic outliers fall in similar range
                real_out_min, real_out_max = (
                    real_outlier_vals.min(),
                    real_outlier_vals.max(),
                )
                synth_in_range = (
                    (synth_outlier_vals >= real_out_min) & (synth_outlier_vals <= real_out_max)
                ).mean()

                overlaps.append(float(synth_in_range))

        return float(np.mean(overlaps))

    def _compute_density_coverage(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """Compute density coverage using histogram overlap."""
        coverages = []

        for col in real_df.columns:
            # Use common bin edges
            combined = pd.concat([real_df[col], synth_df[col]])
            bins = np.histogram_bin_edges(combined.dropna(), bins=50)

            real_hist, _ = np.histogram(real_df[col].dropna(), bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_df[col].dropna(), bins=bins, density=True)

            # Normalize
            real_hist = real_hist / (real_hist.sum() + 1e-10)
            synth_hist = synth_hist / (synth_hist.sum() + 1e-10)

            # Overlap = sum of minimum of each bin
            overlap = np.sum(np.minimum(real_hist, synth_hist))
            coverages.append(overlap)

        return float(np.mean(coverages))

    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using specified method."""
        clean_series = series.dropna()

        if self.outlier_method == "iqr":
            q1 = clean_series.quantile(0.25)
            q3 = clean_series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr
            return (series < lower) | (series > upper)
        # zscore
        z = np.abs(stats.zscore(clean_series))
        outlier_idx = clean_series.index[z > self.zscore_threshold]
        return series.index.isin(outlier_idx)
