"""
k-Anonymity Privacy Metric.

Measures whether synthetic data achieves k-anonymity, meaning each
combination of quasi-identifiers appears at least k times.

Higher k values indicate better privacy protection.
Score is normalized to [0, 1] where 1 = most private.
"""

from typing import Any

import numpy as np
import pandas as pd

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import ReidentificationMetric


@register("privacy.k_anonymity")
class KAnonymityMetric(ReidentificationMetric):
    """
    k-Anonymity Privacy Metric.

    k-Anonymity requires that each combination of quasi-identifier values
    appears in at least k records, making it impossible to uniquely
    identify any individual.

    Process:
    1. Group records by quasi-identifiers
    2. Count group sizes
    3. Find minimum group size (effective k)
    4. Normalize to privacy score

    Interpretation:
        - k = 1: Each record unique → Score = 0.0 (no privacy)
        - k ≥ target_k: Good anonymity → Score approaches 1.0

    References:
        - Sweeney, L. (2002): k-Anonymity: A Model for Protecting Privacy
    """

    name: str = "k_anonymity"
    purpose_tags: set = {"privacy", "dataset_based", "reidentification", "k_anonymity"}

    def __init__(
        self,
        quasi_identifiers: list[str] | None = None,
        target_k: int = 5,
    ):
        """
        Initialize k-Anonymity metric.

        Args:
            quasi_identifiers: Columns to use as quasi-identifiers.
                              If None, uses all categorical columns.
            target_k: Target k value for full privacy score
        """
        super().__init__()
        self.quasi_identifiers = quasi_identifiers
        self.target_k = target_k

    def _get_quasi_identifiers(self) -> list[str]:
        """Return the columns to be used as quasi-identifiers.

        Resolution order (no silent fallback to "first N columns", which would
        produce arbitrary QI sets and meaningless privacy scores):

        1. Explicit ``quasi_identifiers`` passed at construction (filtered to
           the columns actually present in the synthetic data).
        2. Otherwise, all categorical columns detected on the synthetic data.
        3. Otherwise, an empty list — ``compute()`` will surface this as an
           error result instead of fabricating a QI set.
        """
        if self.quasi_identifiers:
            all_cols = set(self._synth_data.columns)
            return [c for c in self.quasi_identifiers if c in all_cols]
        return self._get_categorical_columns() or []

    def _compute_k_anonymity(self, data: pd.DataFrame, qi_cols: list[str]) -> dict[str, Any]:
        """
        Compute k-anonymity statistics for a dataset.

        Returns:
            Dictionary with k-anonymity metrics
        """
        if not qi_cols:
            return {"error": "No quasi-identifiers specified", "k": 0}

        # Create quasi-identifier tuples
        qi_data = data[qi_cols].astype(str)

        # Group by quasi-identifiers and count
        group_sizes = qi_data.groupby(list(qi_cols)).size()

        if len(group_sizes) == 0:
            return {"error": "No groups formed", "k": 0}

        # Calculate statistics
        k_min = int(group_sizes.min())
        k_max = int(group_sizes.max())
        k_mean = float(group_sizes.mean())
        k_median = float(group_sizes.median())

        n_groups = len(group_sizes)
        n_unique = int((group_sizes == 1).sum())
        n_below_target = int((group_sizes < self.target_k).sum())

        return {
            "k_min": k_min,
            "k_max": k_max,
            "k_mean": k_mean,
            "k_median": k_median,
            "n_groups": n_groups,
            "n_unique_records": n_unique,
            "n_groups_below_target": n_below_target,
            "fraction_below_target": n_below_target / n_groups if n_groups > 0 else 0,
        }

    def compute(self) -> MetricResult:
        """
        Compute k-Anonymity privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            qi_cols = self._get_quasi_identifiers()
            if not qi_cols:
                return self._create_error_result("No quasi-identifiers found")

            # Compute k-anonymity for synthetic data
            synth_stats = self._compute_k_anonymity(self._synth_data, qi_cols)

            if "error" in synth_stats:
                return self._create_error_result(synth_stats["error"])

            # Also compute for real data for comparison
            real_stats = self._compute_k_anonymity(self._real_data, qi_cols)

            # Calculate privacy score based on median k (more robust than min)
            k_min = synth_stats["k_min"]
            k_median = synth_stats["k_median"]

            # Continuous score that does NOT saturate at k_median == target_k:
            #   safe_fraction      = 1 - fraction_below_target
            #   depth_factor       = 1 - exp(-k_median / target_k)   ∈ [0, 1)
            #   privacy_score      = safe_fraction * depth_factor
            # Properties:
            #  - all-unique synthetic (k_median <= 1) → score ≈ 0
            #  - k_median == target_k → ≈ 0.632 * safe_fraction
            #  - k_median ≫ target_k continues to grow (no plateau at 1.0)
            #  - any group below target_k drags the score down via safe_fraction
            safe_fraction = 1.0 - synth_stats["fraction_below_target"]
            if self.target_k <= 0:
                depth_factor = 1.0
            else:
                depth_factor = 1.0 - float(np.exp(-k_median / self.target_k))
            privacy_score = float(np.clip(safe_fraction * depth_factor, 0.0, 1.0))

            details = {
                "k_min": k_min,
                "k_median": k_median,
                "k_mean": synth_stats["k_mean"],
                "target_k": self.target_k,
                "privacy_score": float(privacy_score),
                "quasi_identifiers": qi_cols,
                "n_quasi_identifiers": len(qi_cols),
                "synthetic_stats": synth_stats,
                "real_stats": real_stats if "error" not in real_stats else None,
                "interpretation": self._interpret_score(int(k_median)),
            }

            return MetricResult(
                id="privacy.k_anonymity",
                value=float(privacy_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"k-Anonymity computation failed: {str(e)}")

    def _interpret_score(self, k: int) -> str:
        """Provide human-readable interpretation of the k value."""
        if k >= self.target_k:
            return f"Excellent: k={k} meets target of {self.target_k}"
        if k >= 5:
            return f"Good: k={k} provides reasonable anonymity"
        if k >= 2:
            return f"Moderate: k={k} provides limited anonymity"
        return f"Poor: k={k} - records may be uniquely identifiable"
