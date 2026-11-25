"""
l-Diversity Privacy Metric.

Measures whether the synthetic table discloses the sensitive value of real
individuals through equivalence classes with too few distinct sensitive
values, relative to what an independent sample would disclose.

Score is normalized to [0, 1] where 1 = most private.
"""

import numpy as np

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import SyntacticMetric


@register("privacy.l_diversity")
class LDiversityMetric(SyntacticMetric):
    """
    l-Diversity Privacy Metric (relative to a control).

    l-Diversity requires that each equivalence class (records sharing the
    quasi-identifiers) holds at least l distinct sensitive values, so knowing
    someone's quasi-identifiers does not reveal their sensitive value
    (homogeneity attack). Evaluated as an attack on the synthetic table:

    1. For every real record (victim), find the synthetic class with the same
       generalised quasi-identifiers.
    2. The attack succeeds if the class exists, has fewer than ``target_l``
       distinct sensitive values, and its most frequent value is the
       victim's true value.
    3. The same rate on control victims (holdout against the synthetic table,
       or real against the rest of the real table without holdout) is the
       baseline; ``privacy = 1 − excess risk``.

    Sensitive columns: ``sensitive_columns`` or, by default, the target
    (``context["target"]``). Without either the metric returns NaN with an
    explanatory error instead of treating every column as sensitive. Numeric
    sensitive columns are generalised into ``n_bins`` quantile bins. With
    several sensitive columns the score is their mean.

    References:
        - Machanavajjhala et al. (2007): l-Diversity: Privacy Beyond k-Anonymity
        - Giomi et al. (2023): Anonymeter (attribute inference risk)
    """

    name: str = "l_diversity"
    purpose_tags: set = {"privacy", "dataset_based", "syntactic", "l_diversity"}

    def __init__(
        self,
        quasi_identifiers: list[str] | None = None,
        sensitive_columns: list[str] | None = None,
        target_l: int = 3,
        n_bins: int = 5,
    ):
        """
        Initialize l-Diversity metric.

        Args:
            quasi_identifiers: Columns to use as quasi-identifiers. If None,
                every shared column except the target and the sensitive columns.
            sensitive_columns: Columns whose disclosure is assessed. If None,
                the target column from the context.
            target_l: Classes with fewer distinct sensitive values are homogeneous.
            n_bins: Quantile bins used to generalise numeric columns.
        """
        super().__init__()
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_columns = sensitive_columns
        self.target_l = target_l
        self.n_bins = n_bins

    def compute(self) -> MetricResult:
        """
        Compute l-Diversity privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            if self._get_holdout() is None:
                return self._no_holdout_result("l-diversity")
            sensitive_cols = self._resolve_sensitive_columns(self.sensitive_columns)
            if self._get_holdout() is None:  # it lacked the sensitive column(s)
                return self._no_holdout_result("l-diversity")
            if not sensitive_cols:
                return self._create_not_applicable_result(
                    "No sensitive column: set sensitive_columns or data.target"
                )
            qi_cols, dropped = self._resolve_quasi_identifiers(
                self.quasi_identifiers, sensitive_cols
            )
            if not qi_cols:
                return self._create_error_result("No quasi-identifiers found")

            target_l = self.target_l

            def homogeneity_attack(view):
                return (
                    (view["size"] >= 1)
                    & (view["distinct"] < target_l)
                    & (view["own"] > 0)
                    & (view["own"] >= view["top"])
                )

            per_column = {}
            for sens_col in sensitive_cols:
                attack = self._attack_rates(qi_cols, sens_col, self.n_bins, homogeneity_attack)
                train_view = attack.pop("train_view")
                matched = train_view["size"] >= 1
                attack["median_l_matched_classes"] = (
                    float(np.median(train_view["distinct"][matched])) if matched.any() else None
                )
                per_column[sens_col] = attack

            scores = [r["privacy_score"] for r in per_column.values()]
            # Worst column: a leak through one sensitive column is a leak.
            privacy_score = float(np.min(scores))

            details = {
                "privacy_score": privacy_score,
                "excess_risk_signed": float(
                    np.mean([r["excess_risk_signed"] for r in per_column.values()])
                ),
                "min_column_score": float(np.min(scores)),
                "target_l": target_l,
                "sensitive_columns": sensitive_cols,
                "quasi_identifiers": qi_cols,
                "excluded_from_quasi_identifiers": dropped,
                "n_bins": self.n_bins,
                "per_column": per_column,
                "interpretation": self._interpret_score(privacy_score),
                **self._holdout_details(),
            }

            return MetricResult(
                id="privacy.l_diversity",
                value=privacy_score,
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"l-Diversity computation failed: {str(e)}")

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        if score >= 0.9:
            return "Excellent: homogeneous classes disclose no more than for unseen records"
        if score >= 0.7:
            return "Good: limited excess disclosure through homogeneous classes"
        if score >= 0.4:
            return "Moderate: homogeneous classes disclose training records' sensitive values"
        return "Poor: sensitive values of training records are disclosed"
