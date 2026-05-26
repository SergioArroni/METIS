"""
Num↔Num metrics aggregator - Entry point for numeric-numeric metrics.

This module serves as the main entry point for computing all Num↔Num
correlation metrics between pairs of numeric columns:
- Pearson correlation (linear)
- Spearman correlation (rank-based)
- Distance correlation (dCor) - captures non-linear dependencies
- Mutual information (MI) - information-theoretic measure

For individual metrics, see:
- pearson.py: PearsonCorrelationMetric
- spearman.py: SpearmanCorrelationMetric
- dcor.py: DistanceCorrelationMetric
- mi.py: MutualInformationMetric
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ..pair_results import PairMetricResult

_LOGGER = logging.getLogger(__name__)


class NumNumMetrics:
    """
    Compute Num↔Num correlation metrics.

    Compares how well synthetic data preserves correlations between
    pairs of numeric columns. Supports multiple correlation measures
    to capture both linear and non-linear relationships.

    Metrics computed:
        - pearson: Linear correlation coefficient
        - spearman: Rank-based correlation coefficient
        - dcor: Distance correlation (captures non-linear dependencies)
        - mi: Mutual information (discretized)

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
        >>> synth = pd.DataFrame({"a": [1, 2, 3], "b": [2, 5, 7]})
        >>> metrics = NumNumMetrics()
        >>> results = metrics.compute(real, synth)
    """

    # Configuration
    max_samples_dcor: int = 2000  # Limit for O(n²) memory complexity
    mi_bins: int = 10

    def __init__(self):
        self._results: dict[tuple[str, str], dict[str, PairMetricResult]] = {}

    def compute(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, dict[tuple[str, str], PairMetricResult]]:
        """
        Compute all Num↔Num metrics for column pairs.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            pairs: Specific pairs to compute. If None, all numeric pairs.

        Returns:
            Dictionary mapping metric names to dictionaries of
            (col1, col2) -> PairMetricResult
        """
        # Get numeric columns
        real_num = set(real_data.select_dtypes(include=[np.number]).columns)
        synth_num = set(synth_data.select_dtypes(include=[np.number]).columns)
        common_num = sorted(real_num & synth_num)

        if pairs is None:
            pairs = [(c1, c2) for i, c1 in enumerate(common_num) for c2 in common_num[i + 1 :]]

        metrics = {
            "pearson": self._compute_pearson,
            "spearman": self._compute_spearman,
            "dcor": self._compute_dcor,
            "mi": self._compute_mi,
        }

        results = {m: {} for m in metrics}

        for c1, c2 in pairs:
            if c1 not in real_data.columns or c2 not in real_data.columns:
                continue
            if c1 not in synth_data.columns or c2 not in synth_data.columns:
                continue

            # Get clean data
            real_pair = real_data[[c1, c2]].dropna()
            synth_pair = synth_data[[c1, c2]].dropna()

            if len(real_pair) < 10 or len(synth_pair) < 10:
                continue

            r1, r2 = real_pair[c1].values, real_pair[c2].values
            s1, s2 = synth_pair[c1].values, synth_pair[c2].values

            for metric_name, metric_func in metrics.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        real_val = metric_func(r1, r2)
                        synth_val = metric_func(s1, s2)

                    if np.isnan(real_val) or np.isnan(synth_val):
                        continue

                    delta = abs(real_val - synth_val)
                    # Normalize: for correlations, delta in [0, 1] or [0, 2]
                    normalized = 1.0 - min(delta, 1.0)

                    results[metric_name][(c1, c2)] = PairMetricResult(
                        col1=c1,
                        col2=c2,
                        real_value=real_val,
                        synth_value=synth_val,
                        delta=delta,
                        normalized_value=normalized,
                        is_valid=True,
                    )
                except (ValueError, TypeError, ArithmeticError) as exc:
                    _LOGGER.warning(
                        "%s pair (%s, %s) failed: %s: %s",
                        metric_name,
                        c1,
                        c2,
                        type(exc).__name__,
                        exc,
                    )

        return results

    @staticmethod
    def _compute_pearson(x: np.ndarray, y: np.ndarray) -> float:
        """Compute absolute Pearson correlation."""
        return abs(stats.pearsonr(x, y)[0])

    @staticmethod
    def _compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Compute absolute Spearman correlation."""
        return abs(stats.spearmanr(x, y)[0])

    def _compute_dcor(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance correlation with subsampling for large datasets.

        Distance correlation captures non-linear dependencies.
        O(n²) memory complexity - uses subsampling for large datasets.
        """
        n = len(x)
        if n < 4:
            return 0.0

        # Subsample if too large
        if n > self.max_samples_dcor:
            seed = self._context.get("seed", 42)
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, self.max_samples_dcor, replace=False)
            x = x[idx]
            y = y[idx]
            n = self.max_samples_dcor

        # Distance matrices (n×n)
        a = np.abs(x[:, None] - x)
        b = np.abs(y[:, None] - y)

        # Double centering
        a_row = a.mean(axis=1, keepdims=True)
        a_col = a.mean(axis=0, keepdims=True)
        a_mean = a.mean()
        A = a - a_row - a_col + a_mean

        b_row = b.mean(axis=1, keepdims=True)
        b_col = b.mean(axis=0, keepdims=True)
        b_mean = b.mean()
        B = b - b_row - b_col + b_mean

        # Distance covariance and variances
        dcov_sq = (A * B).mean()
        dvar_x = (A * A).mean()
        dvar_y = (B * B).mean()

        if dvar_x * dvar_y <= 0:
            return 0.0

        return float(np.sqrt(dcov_sq / np.sqrt(dvar_x * dvar_y)))

    def _compute_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information (discretized).

        MI measures information-theoretic dependency between variables.
        """
        # Discretize continuous variables
        x_binned = pd.cut(x, bins=self.mi_bins, labels=False, duplicates="drop")
        y_binned = pd.cut(y, bins=self.mi_bins, labels=False, duplicates="drop")

        # Handle NaN from binning
        mask = ~(np.isnan(x_binned) | np.isnan(y_binned))
        if mask.sum() < 10:
            return 0.0

        x_binned = x_binned[mask].astype(int)
        y_binned = y_binned[mask].astype(int)

        # Compute joint and marginal distributions
        contingency = np.histogram2d(x_binned, y_binned, bins=self.mi_bins)[0]
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
