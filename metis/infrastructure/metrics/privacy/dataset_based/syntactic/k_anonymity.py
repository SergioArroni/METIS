"""
k-Anonymity Privacy Metric.

Measures whether the synthetic table lets an adversary single out real
individuals through small equivalence classes of quasi-identifiers, relative
to what an independent sample of the same population would allow.

Score is normalized to [0, 1] where 1 = most private.
"""

from typing import Any

import numpy as np
import pandas as pd

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import SyntacticMetric


@register("privacy.k_anonymity")
class KAnonymityMetric(SyntacticMetric):
    """
    k-Anonymity Privacy Metric (relative to a control).

    k-Anonymity requires that each combination of quasi-identifier values
    appears in at least k records. For a synthetic table the question is
    whether it lets the attacker single out *training* records:

    1. For every real record (victim), find the synthetic equivalence class
       with the same generalised quasi-identifiers.
    2. The victim is exposed if that class exists and has fewer than
       ``target_k`` records (``1 ≤ k < target_k``).
    3. The same rate on control victims (holdout records against the
       synthetic table, or real records against the rest of the real table
       when there is no holdout) is the baseline.
    4. ``privacy = 1 − max(0, (rate − baseline) / (1 − baseline))``.

    Interpretation:
        - a sample independent of the training data → score ≈ 1
        - a copy of the training data with fine quasi-identifiers → score ≈ 0

    The classic synthetic-only statistics (k_min, k_median of the synthetic
    table and of the real one) are kept in ``details`` for reference; they
    are not the score, because with many quasi-identifiers almost every row
    of any 400-row table is unique, independent samples included.

    References:
        - Sweeney, L. (2002): k-Anonymity: A Model for Protecting Privacy
        - Giomi et al. (2023): A Unified Framework for Quantifying Privacy Risk
          in Synthetic Data (Anonymeter, singling-out risk)
    """

    name: str = "k_anonymity"
    purpose_tags: set = {"privacy", "dataset_based", "syntactic", "k_anonymity"}

    def __init__(
        self,
        quasi_identifiers: list[str] | None = None,
        target_k: int = 5,
        n_bins: int = 5,
    ):
        """
        Initialize k-Anonymity metric.

        Args:
            quasi_identifiers: Columns to use as quasi-identifiers. If None,
                every shared column except the target.
            target_k: Classes smaller than this expose their members.
            n_bins: Quantile bins used to generalise numeric quasi-identifiers.
        """
        super().__init__()
        self.quasi_identifiers = quasi_identifiers
        self.target_k = target_k
        self.n_bins = n_bins

    @staticmethod
    def _class_size_stats(keys: np.ndarray, target_k: int) -> dict[str, Any]:
        sizes = pd.Series(keys).value_counts()
        return {
            "k_min": int(sizes.min()),
            "k_median": float(sizes.median()),
            "n_groups": int(len(sizes)),
            "fraction_records_below_target": float(sizes[sizes < target_k].sum() / len(keys)),
        }

    def compute(self) -> MetricResult:
        """
        Compute k-Anonymity privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            if self._get_holdout() is None:
                return self._no_holdout_result("k-anonymity")
            qi_cols, dropped = self._resolve_quasi_identifiers(self.quasi_identifiers, [])
            if not qi_cols:
                return self._create_error_result("No quasi-identifiers found")

            target_k = self.target_k
            attack = self._attack_rates(
                qi_cols,
                None,
                self.n_bins,
                lambda view: (view["size"] >= 1) & (view["size"] < target_k),
            )
            privacy_score = attack["privacy_score"]

            details = {
                "privacy_score": float(privacy_score),
                "epsilon_lower_bound": attack["epsilon_lower_bound"],
                "risk": attack["risk"],
                "excess_risk_signed": attack["excess_risk_signed"],
                "attack_rate_training": attack["attack_rate_training"],
                "attack_rate_control": attack["attack_rate_control"],
                "control": attack["control"],
                "target_k": target_k,
                "quasi_identifiers": qi_cols,
                "excluded_from_quasi_identifiers": dropped,
                "n_bins": self.n_bins,
                "synthetic_stats": self._class_size_stats(
                    self._qi_keys(self._synth_data, qi_cols, self.n_bins), target_k
                ),
                "real_stats": self._class_size_stats(
                    self._qi_keys(self._real_data, qi_cols, self.n_bins), target_k
                ),
                "interpretation": self._interpret_score(
                    privacy_score,
                    attack["attack_rate_training"],
                    attack["attack_rate_control"],
                ),
                **self._holdout_details(),
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

    def _interpret_score(self, score: float, rate: float, control: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        msg = (
            f"{rate:.1%} of training records fall in a synthetic class with k<{self.target_k} "
            f"(control {control:.1%})"
        )
        if score >= 0.9:
            return f"Excellent: {msg}"
        if score >= 0.7:
            return f"Good: {msg}"
        if score >= 0.4:
            return f"Moderate: {msg}"
        return f"Poor: {msg} - training records can be singled out"
