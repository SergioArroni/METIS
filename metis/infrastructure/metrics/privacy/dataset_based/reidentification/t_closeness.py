"""
t-Closeness Privacy Metric.

Extends l-diversity by requiring that the distribution of sensitive
attributes within each equivalence class is close to the global distribution.

Lower distance indicates better privacy protection against skewness attacks.
Score is normalized to [0, 1] where 1 = most private.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.aggregation.stochastic_dominance import fsd_score_for_row
from metis.infrastructure.metrics.registry import register

from ...privacy_base import ReidentificationMetric

_LOGGER = logging.getLogger(__name__)


@register("privacy.t_closeness")
class TClosenessMetric(ReidentificationMetric):
    """
    t-Closeness Privacy Metric.

    t-Closeness requires that the distribution of sensitive attributes
    within each equivalence class differs from the global distribution
    by at most a threshold t.

    This protects against skewness attacks and similarity attacks that
    exploit non-uniform distributions within equivalence classes.

    Process:
    1. Compute global distribution of sensitive attribute
    2. For each equivalence class, compute local distribution
    3. Measure Earth Mover's Distance (EMD) between local and global
    4. Privacy score based on maximum distance

    Interpretation:
        - Distance = 0: Perfect closeness → Score = 1.0
        - Distance ≥ threshold_t: Poor closeness → Score approaches 0.0

    References:
        - Li et al. (2007): t-Closeness: Privacy Beyond k-Anonymity and l-Diversity
    """

    name: str = "t_closeness"
    purpose_tags: set = {"privacy", "dataset_based", "reidentification", "t_closeness"}

    def __init__(
        self,
        quasi_identifiers: list[str] | None = None,
        sensitive_column: str | None = None,
        threshold_t: float = 0.2,
    ):
        """
        Initialize t-Closeness metric.

        Args:
            quasi_identifiers: Columns to use as quasi-identifiers.
            sensitive_column: Column to check for closeness.
            threshold_t: Maximum acceptable distance for full privacy score
        """
        super().__init__()
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_column = sensitive_column
        self.threshold_t = threshold_t

    def _get_quasi_identifiers(self) -> list[str]:
        """Get quasi-identifier columns."""
        if self.quasi_identifiers:
            all_cols = set(self._synth_data.columns)
            return [c for c in self.quasi_identifiers if c in all_cols]
        cat_cols = self._get_categorical_columns()
        # Get sensitive directly to avoid potential issues
        sens = None
        if self.sensitive_column and self.sensitive_column in self._synth_data.columns:
            sens = self.sensitive_column
        else:
            target = self._context.get("target")
            if target and target in self._synth_data.columns:
                sens = target
        return [c for c in cat_cols if c != sens][:5]

    def _get_sensitive_column(self) -> str | None:
        """Get the sensitive column."""
        if self.sensitive_column and self.sensitive_column in self._synth_data.columns:
            return self.sensitive_column

        # Check for target column in context
        target = self._context.get("target")
        if target and target in self._synth_data.columns:
            return target

        # Fallback: use first categorical column not in QI
        cat_cols = self._get_categorical_columns()
        if cat_cols:
            # Get QI without calling _get_quasi_identifiers to avoid issues
            qi_set = set()
            if self.quasi_identifiers:
                qi_set = set(self.quasi_identifiers)
            remaining = [c for c in cat_cols if c not in qi_set]
            if remaining:
                return remaining[0]

        return None

    def _compute_emd_categorical(
        self,
        global_dist: dict[str, float],
        local_dist: dict[str, float],
    ) -> float:
        """
        Compute Earth Mover's Distance for categorical distributions.

        For categorical, uses total variation distance (half L1).
        """
        all_values = set(global_dist.keys()) | set(local_dist.keys())

        total_diff = 0.0
        for val in all_values:
            p = global_dist.get(val, 0.0)
            q = local_dist.get(val, 0.0)
            total_diff += abs(p - q)

        # EMD for categorical = 0.5 * L1 distance
        return total_diff / 2.0

    def _compute_emd_numerical(
        self,
        global_values: np.ndarray,
        local_values: np.ndarray,
    ) -> float:
        """
        Compute Earth Mover's Distance for numerical distributions.

        Uses Wasserstein distance (1D EMD).

        Returns NaN when EMD is undefined (empty local sample or scipy failure)
        so callers can mark the group as skipped instead of inflating the
        privacy score with a silent ``1.0`` fallback.
        """
        if len(local_values) == 0:
            return float("nan")

        try:
            distance = stats.wasserstein_distance(global_values, local_values)
        except (ValueError, RuntimeError) as exc:
            _LOGGER.warning("t_closeness EMD failed (numerical): %s", exc)
            return float("nan")
        data_range = np.ptp(global_values)
        if data_range > 0:
            return float(min(1.0, distance / data_range))
        return 0.0

    def _compute_t_closeness(
        self,
        data: pd.DataFrame,
        qi_cols: list[str],
        sens_col: str,
    ) -> dict[str, Any]:
        """
        Compute t-closeness statistics for a dataset.

        Returns:
            Dictionary with t-closeness metrics
        """
        if not qi_cols or not sens_col:
            return {"error": "Missing quasi-identifiers or sensitive column", "t": 1.0}

        sens_data = data[sens_col]
        is_numerical = pd.api.types.is_numeric_dtype(sens_data)

        # Compute global distribution
        if is_numerical:
            global_values = sens_data.dropna().values
        else:
            global_counts = sens_data.value_counts(normalize=True)
            global_dist = global_counts.to_dict()

        # Group by quasi-identifiers
        grouped = data.groupby(list(qi_cols))

        distances = []
        group_stats = []

        for _group_key, group_data in grouped:
            local_sens = group_data[sens_col]

            if is_numerical:
                local_values = local_sens.dropna().values
                distance = self._compute_emd_numerical(global_values, local_values)
            else:
                local_counts = local_sens.value_counts(normalize=True)
                local_dist = local_counts.to_dict()
                distance = self._compute_emd_categorical(global_dist, local_dist)

            if np.isnan(distance):
                # Skip undefined groups; keep counter for diagnostics.
                continue
            distances.append(distance)
            group_stats.append(
                {
                    "group_size": len(group_data),
                    "distance": float(distance),
                }
            )

        if not distances:
            return {"error": "No groups formed", "t": 1.0}

        distances = np.array(distances)

        return {
            "t_max": float(distances.max()),
            "t_mean": float(distances.mean()),
            "t_median": float(np.median(distances)),
            "n_groups": len(distances),
            "n_groups_above_threshold": int((distances > self.threshold_t).sum()),
            "fraction_above_threshold": float((distances > self.threshold_t).mean()),
            "is_numerical": is_numerical,
        }

    def _get_all_sensitive_columns(self) -> list[str]:
        """Get all columns to test as sensitive (categorical + numeric)."""
        all_cols = list(self._synth_data.columns)
        # Exclude QI columns
        qi_set = set()
        if self.quasi_identifiers:
            qi_set = set(self.quasi_identifiers)
        return [c for c in all_cols if c not in qi_set]

    def compute(self) -> MetricResult:
        """
        Compute t-Closeness privacy metric for all columns.

        Tests each column as sensitive and aggregates results.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            qi_cols = self._get_quasi_identifiers()

            if not qi_cols:
                return self._create_error_result("No quasi-identifiers found")

            # Get all columns to test as sensitive
            if self.sensitive_column:
                # Single column mode
                sensitive_cols = [self.sensitive_column]
            else:
                # Test all columns
                sensitive_cols = self._get_all_sensitive_columns()

            if not sensitive_cols:
                return self._create_error_result("No sensitive columns found")

            # Compute t-closeness for each column
            column_results = {}
            t_max_values = []

            for sens_col in sensitive_cols:
                if sens_col not in self._synth_data.columns:
                    continue

                synth_stats = self._compute_t_closeness(self._synth_data, qi_cols, sens_col)

                if "error" in synth_stats:
                    column_results[sens_col] = {
                        "skipped": True,
                        "reason": synth_stats["error"],
                    }
                    continue

                real_stats = self._compute_t_closeness(self._real_data, qi_cols, sens_col)

                t_max = synth_stats["t_max"]
                t_max_values.append(t_max)

                # Column score uses both the magnitude of the worst group EMD
                # AND the fraction of groups breaching the configured
                # ``threshold_t``. Previously ``threshold_t`` was only echoed
                # in the report; now it actively penalises the score:
                #   col_score = (1 - t_max) * (1 - fraction_above_threshold)
                # This keeps the [0, 1] range (1 = most private) while making
                # ``threshold_t`` a real configurable knob.
                frac_breach = float(synth_stats.get("fraction_above_threshold", 0.0))
                col_score = float(np.clip((1.0 - t_max) * (1.0 - frac_breach), 0.0, 1.0))

                column_results[sens_col] = {
                    "t_max": t_max,
                    "t_mean": synth_stats["t_mean"],
                    "t_median": synth_stats["t_median"],
                    "n_groups": synth_stats["n_groups"],
                    "fraction_above_threshold": synth_stats["fraction_above_threshold"],
                    "is_numerical": synth_stats["is_numerical"],
                    "normalized_value": col_score,
                    "real_t_max": (real_stats.get("t_max") if "error" not in real_stats else None),
                }

            if not t_max_values:
                return self._create_error_result("No valid columns to compute t-closeness")

            # Collect per-column scores
            col_scores = [
                column_results[col]["normalized_value"]
                for col in column_results
                if "normalized_value" in column_results.get(col, {})
            ]

            # Aggregate using FSD (First-Order Stochastic Dominance)
            privacy_score = float(fsd_score_for_row(np.array(col_scores))) if col_scores else 0.0

            overall_t_max = max(t_max_values)
            mean_t = float(np.mean(t_max_values))

            # Summary stats
            valid_count = len(t_max_values)
            skipped_count = len([r for r in column_results.values() if r.get("skipped")])

            details = {
                "threshold_t": self.threshold_t,
                "quasi_identifiers": qi_cols,
                "overall_t_max": overall_t_max,
                "mean_t": mean_t,
                "per_column": column_results,
                "_summary": {
                    "total_columns": len(sensitive_cols),
                    "valid_columns": valid_count,
                    "skipped_columns": skipped_count,
                    "columns_above_threshold": int(
                        sum(1 for t in t_max_values if t > self.threshold_t)
                    ),
                },
                "interpretation": self._interpret_score(mean_t),
            }

            return MetricResult(
                id="privacy.t_closeness",
                value=float(privacy_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"t-Closeness computation failed: {str(e)}")

    def _interpret_score(self, t: float) -> str:
        """Provide human-readable interpretation of the t value."""
        if t <= self.threshold_t:
            return f"Excellent: t={t:.3f} meets threshold of {self.threshold_t}"
        if t <= 0.3:
            return f"Good: t={t:.3f} provides reasonable closeness"
        if t <= 0.5:
            return f"Moderate: t={t:.3f} shows some distribution skew"
        return f"Poor: t={t:.3f} - significant distribution differences in groups"
