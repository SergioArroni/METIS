"""
Num↔Cat metrics aggregator - Entry point for numeric-categorical metrics.

This module serves as the main entry point for computing all Num↔Cat
association metrics between numeric and categorical columns:
- Point-biserial correlation (for binary categories)
- Eta-squared (effect size from ANOVA)
- Kruskal-Wallis epsilon-squared (non-parametric)

For individual metrics, see:
- point_biserial.py: PointBiserialMetric
- eta_squared.py: EtaSquaredMetric
- kruskal_epsilon.py: KruskalEpsilonMetric
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pointbiserialr

from ..pair_results import PairMetricResult

_LOGGER = logging.getLogger(__name__)


class NumCatMetrics:
    """
    Compute Num↔Cat relationship metrics.

    Compares how well synthetic data preserves associations between
    numeric and categorical columns. Uses different metrics depending
    on the nature of the categorical variable.

    Metrics computed:
        - point_biserial: Point-biserial correlation (binary categories only)
        - eta_squared: Effect size from ANOVA
        - kruskal_epsilon: Effect size from Kruskal-Wallis test

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"age": [25, 30, 35], "gender": ["M", "F", "M"]})
        >>> synth = pd.DataFrame({"age": [26, 31, 34], "gender": ["M", "F", "M"]})
        >>> metrics = NumCatMetrics()
        >>> results = metrics.compute(real, synth)
    """

    def compute(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, dict[tuple[str, str], PairMetricResult]]:
        """
        Compute Num↔Cat metrics.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            pairs: list of (numeric_col, categorical_col) tuples.
                   If None, all numeric-categorical pairs.

        Returns:
            Dictionary mapping metric names to dictionaries of
            (numeric_col, categorical_col) -> PairMetricResult
        """
        # Identify column types
        real_num = set(real_data.select_dtypes(include=[np.number]).columns)
        real_cat = set(real_data.select_dtypes(include=["object", "category"]).columns)
        synth_num = set(synth_data.select_dtypes(include=[np.number]).columns)
        synth_cat = set(synth_data.select_dtypes(include=["object", "category"]).columns)

        common_num = sorted(real_num & synth_num)
        common_cat = sorted(real_cat & synth_cat)

        if pairs is None:
            pairs = [(n, c) for n in common_num for c in common_cat]

        results = {
            "point_biserial": {},
            "eta_squared": {},
            "kruskal_epsilon": {},
        }

        for num_col, cat_col in pairs:
            if num_col not in common_num or cat_col not in common_cat:
                continue

            real_pair = real_data[[num_col, cat_col]].dropna()
            synth_pair = synth_data[[num_col, cat_col]].dropna()

            if len(real_pair) < 20 or len(synth_pair) < 20:
                continue

            # Point-biserial (for binary categories)
            self._compute_point_biserial(real_pair, synth_pair, num_col, cat_col, results)

            # Eta-squared (from ANOVA)
            self._compute_eta_squared(real_pair, synth_pair, num_col, cat_col, results)

            # Kruskal-Wallis epsilon-squared
            self._compute_kruskal_epsilon(real_pair, synth_pair, num_col, cat_col, results)

        return results

    def _compute_point_biserial(
        self,
        real_pair: pd.DataFrame,
        synth_pair: pd.DataFrame,
        num_col: str,
        cat_col: str,
        results: dict[str, dict[tuple[str, str], PairMetricResult]],
    ) -> None:
        """Compute point-biserial correlation for binary categories."""
        real_cats = real_pair[cat_col].unique()
        synth_cats = synth_pair[cat_col].unique()

        if len(real_cats) == 2 and len(synth_cats) == 2:
            try:
                real_binary = (real_pair[cat_col] == real_cats[0]).astype(int)
                synth_binary = (synth_pair[cat_col] == synth_cats[0]).astype(int)

                r_pb, _ = pointbiserialr(real_binary, real_pair[num_col])
                s_pb, _ = pointbiserialr(synth_binary, synth_pair[num_col])

                delta = abs(abs(r_pb) - abs(s_pb))
                results["point_biserial"][(num_col, cat_col)] = PairMetricResult(
                    col1=num_col,
                    col2=cat_col,
                    real_value=abs(r_pb),
                    synth_value=abs(s_pb),
                    delta=delta,
                    normalized_value=1.0 - min(delta, 1.0),
                    is_valid=True,
                )
            except (ValueError, TypeError, ArithmeticError) as exc:
                _LOGGER.warning(
                    "point-biserial pair (%s, %s) failed: %s: %s",
                    num_col,
                    cat_col,
                    type(exc).__name__,
                    exc,
                )

    def _compute_eta_squared_value(self, groups: list[np.ndarray]) -> float:
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

    def _compute_eta_squared(
        self,
        real_pair: pd.DataFrame,
        synth_pair: pd.DataFrame,
        num_col: str,
        cat_col: str,
        results: dict[str, dict[tuple[str, str], PairMetricResult]],
    ) -> None:
        """Compute eta-squared from ANOVA."""
        try:
            real_groups = [g[num_col].values for _, g in real_pair.groupby(cat_col) if len(g) > 1]
            synth_groups = [g[num_col].values for _, g in synth_pair.groupby(cat_col) if len(g) > 1]

            if len(real_groups) >= 2 and len(synth_groups) >= 2:
                r_eta = self._compute_eta_squared_value(real_groups)
                s_eta = self._compute_eta_squared_value(synth_groups)
                delta = abs(r_eta - s_eta)

                results["eta_squared"][(num_col, cat_col)] = PairMetricResult(
                    col1=num_col,
                    col2=cat_col,
                    real_value=r_eta,
                    synth_value=s_eta,
                    delta=delta,
                    normalized_value=1.0 - min(delta, 1.0),
                    is_valid=True,
                )
        except Exception as e:
            logging.getLogger(__name__).debug(
                "eta_squared skipped for (%s, %s): %s", num_col, cat_col, e
            )

    def _compute_kruskal_epsilon(
        self,
        real_pair: pd.DataFrame,
        synth_pair: pd.DataFrame,
        num_col: str,
        cat_col: str,
        results: dict[str, dict[tuple[str, str], PairMetricResult]],
    ) -> None:
        """Compute Kruskal-Wallis epsilon-squared."""
        try:
            real_groups = [g[num_col].values for _, g in real_pair.groupby(cat_col) if len(g) > 1]
            synth_groups = [g[num_col].values for _, g in synth_pair.groupby(cat_col) if len(g) > 1]

            if len(real_groups) >= 2 and len(synth_groups) >= 2:
                r_stat, _ = kruskal(*real_groups)
                s_stat, _ = kruskal(*synth_groups)

                # Epsilon-squared = H / (n - 1)
                r_eps = r_stat / (len(real_pair) - 1)
                s_eps = s_stat / (len(synth_pair) - 1)
                delta = abs(r_eps - s_eps)

                results["kruskal_epsilon"][(num_col, cat_col)] = PairMetricResult(
                    col1=num_col,
                    col2=cat_col,
                    real_value=r_eps,
                    synth_value=s_eps,
                    delta=delta,
                    normalized_value=1.0 - min(delta, 1.0),
                    is_valid=True,
                )
        except Exception as e:
            logging.getLogger(__name__).debug(
                "kruskal_epsilon skipped for (%s, %s): %s", num_col, cat_col, e
            )
