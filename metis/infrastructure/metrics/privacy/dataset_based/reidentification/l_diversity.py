"""
l-Diversity Privacy Metric.

Extends k-anonymity by requiring that each equivalence class has
at least l "well-represented" values for sensitive attributes.

Higher l values indicate better privacy protection against attribute disclosure.
Score is normalized to [0, 1] where 1 = most private.
"""

from typing import Any

import numpy as np
import pandas as pd

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.aggregation.stochastic_dominance import fsd_score_for_row
from metis.infrastructure.metrics.registry import register

from ...privacy_base import ReidentificationMetric


@register("privacy.l_diversity")
class LDiversityMetric(ReidentificationMetric):
    """
    l-Diversity Privacy Metric.

    l-Diversity requires that each equivalence class (group of records
    with the same quasi-identifiers) has at least l distinct values
    for each sensitive attribute.

    This protects against attribute disclosure attacks where an attacker
    knows someone is in a group but wants to infer their sensitive value.

    Process:
    1. Group records by quasi-identifiers
    2. For each group, count distinct sensitive values
    3. Find minimum diversity across groups
    4. Normalize to privacy score

    Interpretation:
        - l = 1: All records in group have same sensitive value → Score = 0.0
        - l ≥ target_l: Good diversity → Score approaches 1.0

    References:
        - Machanavajjhala et al. (2007): l-Diversity: Privacy Beyond k-Anonymity
    """

    name: str = "l_diversity"
    purpose_tags: set = {"privacy", "dataset_based", "reidentification", "l_diversity"}

    def __init__(
        self,
        quasi_identifiers: list[str] | None = None,
        sensitive_columns: list[str] | None = None,
        target_l: int = 3,
    ):
        """
        Initialize l-Diversity metric.

        Args:
            quasi_identifiers: Columns to use as quasi-identifiers.
            sensitive_columns: Columns to check for diversity.
                              If None, uses the target column if available.
            target_l: Target l value for full privacy score
        """
        super().__init__()
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_columns = sensitive_columns
        self.target_l = target_l

    def _get_quasi_identifiers(self) -> list[str]:
        """Get quasi-identifier columns."""
        if self.quasi_identifiers:
            all_cols = set(self._synth_data.columns)
            return [c for c in self.quasi_identifiers if c in all_cols]
        # Default to categorical columns (limit to 5)
        cat_cols = self._get_categorical_columns()
        return cat_cols[:5]

    def _get_all_sensitive_columns(self) -> list[str]:
        """Get all columns to test as sensitive."""
        all_cols = list(self._synth_data.columns)
        # Exclude QI columns
        qi_set = set()
        if self.quasi_identifiers:
            qi_set = set(self.quasi_identifiers)
        return [c for c in all_cols if c not in qi_set]

    def _compute_l_diversity_single(
        self,
        data: pd.DataFrame,
        qi_cols: list[str],
        sens_col: str,
    ) -> dict[str, Any]:
        """
        Compute l-diversity statistics for a single sensitive column.

        Returns:
            Dictionary with l-diversity metrics for this column
        """
        try:
            # Group by quasi-identifiers
            group_diversity = data.groupby(list(qi_cols))[sens_col].nunique().values

            if len(group_diversity) == 0:
                return {"error": "No groups formed", "l": 0}

            l_min = int(group_diversity.min())
            l_max = int(group_diversity.max())
            l_mean = float(group_diversity.mean())

            n_groups = len(group_diversity)
            n_below_target = int((group_diversity < self.target_l).sum())

            return {
                "l_min": l_min,
                "l_max": l_max,
                "l_mean": l_mean,
                "n_groups": n_groups,
                "n_groups_below_target": n_below_target,
                "fraction_below_target": (n_below_target / n_groups if n_groups > 0 else 0),
            }
        except Exception as e:
            return {"error": str(e), "l": 0}

    def compute(self) -> MetricResult:
        """
        Compute l-Diversity privacy metric for all columns.

        Tests each column as sensitive and aggregates results.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            qi_cols = self._get_quasi_identifiers()

            if not qi_cols:
                return self._create_error_result("No quasi-identifiers found")

            # Get all columns to test as sensitive
            if self.sensitive_columns:
                sensitive_cols = [
                    c for c in self.sensitive_columns if c in self._synth_data.columns
                ]
            else:
                sensitive_cols = self._get_all_sensitive_columns()

            if not sensitive_cols:
                return self._create_error_result("No sensitive columns found")

            # Compute l-diversity for each column
            column_results = {}
            l_min_values = []
            valid_columns = 0
            skipped_columns = 0

            for sens_col in sensitive_cols:
                if sens_col not in self._synth_data.columns:
                    continue
                if sens_col in qi_cols:
                    # Skip QI columns
                    continue

                synth_stats = self._compute_l_diversity_single(self._synth_data, qi_cols, sens_col)

                if "error" in synth_stats:
                    column_results[sens_col] = {
                        "skipped": True,
                        "reason": synth_stats["error"],
                    }
                    skipped_columns += 1
                    continue

                real_stats = self._compute_l_diversity_single(self._real_data, qi_cols, sens_col)

                l_min = synth_stats["l_min"]
                l_min_values.append(l_min)
                valid_columns += 1

                # Continuous column score (mirrors the k-anonymity rework):
                #   safe_fraction * (1 - exp(-l_min / target_l))
                # Avoids the previous saturation at l_min == target_l, so very
                # diverse synthetic distributions (e.g. l_min ≫ target_l) are
                # still discriminated from "barely passing" ones.
                safe_fraction = 1.0 - synth_stats.get("fraction_below_target", 0.0)
                if self.target_l <= 0:
                    depth_factor = 1.0
                else:
                    depth_factor = 1.0 - float(np.exp(-l_min / self.target_l))
                col_score = float(np.clip(safe_fraction * depth_factor, 0.0, 1.0))

                column_results[sens_col] = {
                    "l_min": l_min,
                    "l_max": synth_stats["l_max"],
                    "l_mean": synth_stats["l_mean"],
                    "n_groups": synth_stats["n_groups"],
                    "fraction_below_target": synth_stats["fraction_below_target"],
                    "normalized_value": col_score,
                    "real_l_min": (real_stats.get("l_min") if "error" not in real_stats else None),
                }

            if not l_min_values:
                return self._create_error_result("No valid columns to analyze")

            # Collect per-column scores
            col_scores = [
                column_results[col]["normalized_value"]
                for col in column_results
                if col != "_summary" and "normalized_value" in column_results[col]
            ]

            # Aggregate using FSD (First-Order Stochastic Dominance)
            privacy_score = float(fsd_score_for_row(np.array(col_scores))) if col_scores else 0.0

            # Also keep track of min/mean l values for reporting
            overall_l_min = min(l_min_values)
            mean_l_min = float(np.mean(l_min_values))
            median_l_min = float(np.median(l_min_values))

            # Summary
            column_results["_summary"] = {
                "total_columns": len(sensitive_cols),
                "valid_columns": valid_columns,
                "skipped_columns": skipped_columns,
                "coverage_pct": (
                    round(100 * valid_columns / len(sensitive_cols), 1) if sensitive_cols else 0
                ),
            }

            details = {
                "l_min": overall_l_min,
                "l_median": median_l_min,
                "l_mean": mean_l_min,
                "target_l": self.target_l,
                "privacy_score": float(privacy_score),
                "quasi_identifiers": qi_cols,
                "columns_analyzed": valid_columns,
                "per_column": column_results,
                "interpretation": self._interpret_score(int(median_l_min)),
            }

            return MetricResult(
                id="privacy.l_diversity",
                value=float(privacy_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"l-Diversity computation failed: {str(e)}")

    def _interpret_score(self, l_value: int) -> str:
        """Provide human-readable interpretation of the l value."""
        if l_value >= self.target_l:
            return f"Excellent: l={l_value} meets target of {self.target_l}"
        if l_value >= 3:
            return f"Good: l={l_value} provides reasonable diversity"
        if l_value >= 2:
            return f"Moderate: l={l_value} provides limited diversity"
        return f"Poor: l={l_value} - sensitive values may be homogeneous"
