"""
Cat↔Cat metrics aggregator - Entry point for categorical-categorical metrics.

This module serves as the main entry point for computing all Cat↔Cat
association metrics between pairs of categorical columns:
- Cramér's V (symmetric association measure)
- Theil's U (asymmetric uncertainty coefficient)
- Chi-squared statistic (normalized)

For individual metrics, see:
- cramers_v.py: CramersVMetric
- theils_u.py: TheilsUMetric
- chi2_stat.py: Chi2StatMetric
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ..pair_results import PairMetricResult

_LOGGER = logging.getLogger(__name__)


class CatCatMetrics:
    """
    Compute Cat↔Cat association metrics.

    Compares how well synthetic data preserves associations between
    pairs of categorical columns using contingency table analysis.

    Metrics computed:
        - cramers_v: Cramér's V (symmetric association measure)
        - theils_u: Theil's U (asymmetric uncertainty coefficient)
        - chi2_stat: Normalized chi-squared statistic

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"gender": ["M", "F", "M"], "status": ["A", "B", "A"]})
        >>> synth = pd.DataFrame({"gender": ["M", "F", "F"], "status": ["A", "B", "B"]})
        >>> metrics = CatCatMetrics()
        >>> results = metrics.compute(real, synth)
    """

    def _cramers_v(self, contingency: np.ndarray) -> float:
        """
        Compute Cramér's V from a contingency table.

        Args:
            contingency: Contingency table as 2D numpy array

        Returns:
            Cramér's V value in [0, 1]
        """
        chi2 = chi2_contingency(contingency)[0]
        n = contingency.sum()
        min_dim = min(contingency.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))

    def _theils_u(self, contingency: np.ndarray) -> float:
        """
        Compute Theil's U (uncertainty coefficient) from a contingency table.

        Formula: U(Y|X) = (H(Y) - H(Y|X)) / H(Y)

        Args:
            contingency: Contingency table as 2D numpy array

        Returns:
            Theil's U value in [0, 1]
        """
        pxy = contingency / contingency.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # H(Y)
        hy = -np.sum(py[py > 0] * np.log(py[py > 0]))

        # H(Y|X) = sum_x p(x) * H(Y|X=x)
        hy_given_x = 0
        for i in range(len(px)):
            if px[i] > 0:
                py_given_x = pxy[i, :] / px[i]
                py_given_x = py_given_x[py_given_x > 0]
                hy_given_x -= px[i] * np.sum(py_given_x * np.log(py_given_x))

        if hy == 0:
            return 0.0
        return float((hy - hy_given_x) / hy)

    def compute(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, dict[tuple[str, str], PairMetricResult]]:
        """
        Compute Cat↔Cat metrics.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            pairs: Specific pairs to compute. If None, all categorical pairs.

        Returns:
            Dictionary mapping metric names to dictionaries of
            (col1, col2) -> PairMetricResult
        """
        real_cat = set(real_data.select_dtypes(include=["object", "category"]).columns)
        synth_cat = set(synth_data.select_dtypes(include=["object", "category"]).columns)
        common_cat = sorted(real_cat & synth_cat)

        if pairs is None:
            pairs = [(c1, c2) for i, c1 in enumerate(common_cat) for c2 in common_cat[i + 1 :]]

        results = {
            "cramers_v": {},
            "theils_u": {},
            "chi2_stat": {},
        }

        for c1, c2 in pairs:
            if c1 not in common_cat or c2 not in common_cat:
                continue

            real_pair = real_data[[c1, c2]].dropna()
            synth_pair = synth_data[[c1, c2]].dropna()

            if len(real_pair) < 20 or len(synth_pair) < 20:
                continue

            try:
                real_cont = pd.crosstab(real_pair[c1], real_pair[c2]).values
                synth_cont = pd.crosstab(synth_pair[c1], synth_pair[c2]).values

                if real_cont.size < 4 or synth_cont.size < 4:
                    continue

                # Cramér's V
                self._compute_cramers_v_result(real_cont, synth_cont, c1, c2, results)

                # Theil's U
                self._compute_theils_u_result(real_cont, synth_cont, c1, c2, results)

                # Chi-squared (normalized)
                self._compute_chi2_stat(
                    real_cont, synth_cont, real_pair, synth_pair, c1, c2, results
                )

            except (ValueError, TypeError, ArithmeticError) as exc:
                # Surface the failure in logs so silently-skipped pairs do
                # not bias the aggregated score downstream.
                _LOGGER.warning(
                    "cat-cat pair (%s, %s) failed: %s: %s",
                    c1,
                    c2,
                    type(exc).__name__,
                    exc,
                )

        return results

    def _compute_cramers_v_result(
        self,
        real_cont: np.ndarray,
        synth_cont: np.ndarray,
        c1: str,
        c2: str,
        results: dict[str, dict[tuple[str, str], PairMetricResult]],
    ) -> None:
        """Compute Cramér's V metric result."""
        try:
            r_cv = self._cramers_v(real_cont)
            s_cv = self._cramers_v(synth_cont)
            delta = abs(r_cv - s_cv)
            results["cramers_v"][(c1, c2)] = PairMetricResult(
                col1=c1,
                col2=c2,
                real_value=r_cv,
                synth_value=s_cv,
                delta=delta,
                normalized_value=1.0 - min(delta, 1.0),
                is_valid=True,
            )
        except Exception as e:
            logging.getLogger(__name__).debug("cramers_v skipped for (%s, %s): %s", c1, c2, e)

    def _compute_theils_u_result(
        self,
        real_cont: np.ndarray,
        synth_cont: np.ndarray,
        c1: str,
        c2: str,
        results: dict[str, dict[tuple[str, str], PairMetricResult]],
    ) -> None:
        """Compute Theil's U metric result."""
        try:
            r_tu = self._theils_u(real_cont)
            s_tu = self._theils_u(synth_cont)
            delta = abs(r_tu - s_tu)
            results["theils_u"][(c1, c2)] = PairMetricResult(
                col1=c1,
                col2=c2,
                real_value=r_tu,
                synth_value=s_tu,
                delta=delta,
                normalized_value=1.0 - min(delta, 1.0),
                is_valid=True,
            )
        except Exception as e:
            logging.getLogger(__name__).debug("theils_u skipped for (%s, %s): %s", c1, c2, e)

    def _compute_chi2_stat(
        self,
        real_cont: np.ndarray,
        synth_cont: np.ndarray,
        real_pair: pd.DataFrame,
        synth_pair: pd.DataFrame,
        c1: str,
        c2: str,
        results: dict[str, dict[tuple[str, str], PairMetricResult]],
    ) -> None:
        """Compute normalized chi-squared statistic."""
        try:
            r_chi2 = chi2_contingency(real_cont)[0] / len(real_pair)
            s_chi2 = chi2_contingency(synth_cont)[0] / len(synth_pair)
            delta = abs(r_chi2 - s_chi2)
            results["chi2_stat"][(c1, c2)] = PairMetricResult(
                col1=c1,
                col2=c2,
                real_value=r_chi2,
                synth_value=s_chi2,
                delta=delta,
                normalized_value=1.0 - min(delta, 1.0),
                is_valid=True,
            )
        except Exception as e:
            logging.getLogger(__name__).debug("chi2_stat skipped for (%s, %s): %s", c1, c2, e)
