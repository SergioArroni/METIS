"""
t-Closeness Privacy Metric.

Measures whether the synthetic table discloses information about real
individuals' sensitive values through equivalence classes whose sensitive
distribution is far from the global one (skewness attack), relative to what
an independent sample would disclose.

Score is normalized to [0, 1] where 1 = most private.
"""

import numpy as np

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import SyntacticMetric


@register("privacy.t_closeness")
class TClosenessMetric(SyntacticMetric):
    """
    t-Closeness Privacy Metric (relative to a control).

    t-Closeness requires that the sensitive distribution within each
    equivalence class stays within distance t of the global distribution.
    Evaluated as an attack on the synthetic table:

    1. For every real record (victim), find the synthetic class with the same
       generalised quasi-identifiers.
    2. The attack succeeds if the class exists, its sensitive distribution is
       farther than ``threshold_t`` from the synthetic global distribution
       (Earth Mover's Distance as in Li et al.: ordered ground distance for
       numeric and ordinal columns, equal distance — the total variation —
       for nominal ones; see ``privacy_base._emd``), and it
       puts more mass on the victim's true value than the global distribution
       does (the class reveals something about the victim).
    3. The same rate on control victims is the baseline;
       ``privacy = 1 − excess risk``.

    Sensitive columns: ``sensitive_columns``/``sensitive_column`` or, by
    default, the target from ``context["target"]``; they are never
    quasi-identifiers (formerly every quasi-identifier was also scored as
    sensitive against itself). Numeric sensitive columns are generalised into
    ``n_bins`` quantile bins.

    Known limit of t-closeness (Cao & Karras 2012, PVLDB 5(11)): the bound is
    on an absolute, cumulative difference, so a class made only of a
    frequent value is never farther than ``1 − p`` from the table.  With a
    value above ``1 − threshold_t`` (e.g. 84 % "No" in a binary target and
    t = 0.2) the attack can only succeed on victims of the other values, and
    an exact copy scores about ``1 − (share of the other values)`` instead
    of 0.  The calibration anchors the copy at 0, but the raw value reads as
    private.

    References:
        - Li et al. (2007): t-Closeness: Privacy Beyond k-Anonymity and l-Diversity
        - Giomi et al. (2023): Anonymeter (attribute inference risk)
    """

    name: str = "t_closeness"
    purpose_tags: set = {"privacy", "dataset_based", "syntactic", "t_closeness"}

    def __init__(
        self,
        quasi_identifiers: list[str] | None = None,
        sensitive_column: str | None = None,
        threshold_t: float = 0.2,
        sensitive_columns: list[str] | None = None,
        n_bins: int = 5,
    ):
        """
        Initialize t-Closeness metric.

        Args:
            quasi_identifiers: Columns to use as quasi-identifiers. If None,
                every shared column except the target and the sensitive columns.
            sensitive_column: Single sensitive column (kept for compatibility).
            threshold_t: Classes farther than this from the global
                distribution violate t-closeness.
            sensitive_columns: Sensitive columns; takes precedence over
                ``sensitive_column``. If both are None, the context target.
            n_bins: Quantile bins used to generalise numeric columns.
        """
        super().__init__()
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_column = sensitive_column
        self.sensitive_columns = sensitive_columns
        self.threshold_t = threshold_t
        self.n_bins = n_bins

    def compute(self) -> MetricResult:
        """
        Compute t-Closeness privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            if self._get_holdout() is None:
                return self._no_holdout_result("t-closeness")
            explicit = self.sensitive_columns or (
                [self.sensitive_column] if self.sensitive_column else None
            )
            sensitive_cols = self._resolve_sensitive_columns(explicit)
            if self._get_holdout() is None:  # it lacked the sensitive column(s)
                return self._no_holdout_result("t-closeness")
            if not sensitive_cols:
                return self._create_not_applicable_result(
                    "No sensitive column: set sensitive_columns or data.target"
                )
            qi_cols, dropped = self._resolve_quasi_identifiers(
                self.quasi_identifiers, sensitive_cols
            )
            if not qi_cols:
                return self._create_error_result("No quasi-identifiers found")

            threshold_t = self.threshold_t

            def skewness_attack(view):
                return (
                    (view["size"] >= 1)
                    & (view["emd"] > threshold_t)
                    & (view["p_own"] > view["g_own"])
                )

            per_column = {}
            for sens_col in sensitive_cols:
                attack = self._attack_rates(qi_cols, sens_col, self.n_bins, skewness_attack)
                train_view = attack.pop("train_view")
                matched = train_view["size"] >= 1
                attack["median_t_matched_classes"] = (
                    float(np.median(train_view["emd"][matched])) if matched.any() else None
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
                "threshold_t": threshold_t,
                "sensitive_columns": sensitive_cols,
                "quasi_identifiers": qi_cols,
                "excluded_from_quasi_identifiers": dropped,
                "n_bins": self.n_bins,
                "ground_distance": {c: r["ground_distance"] for c, r in per_column.items()},
                "per_column": per_column,
                "interpretation": self._interpret_score(privacy_score),
                **self._holdout_details(),
            }

            return MetricResult(
                id="privacy.t_closeness",
                value=privacy_score,
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"t-Closeness computation failed: {str(e)}")

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        if score >= 0.9:
            return "Excellent: skewed classes disclose no more than for unseen records"
        if score >= 0.7:
            return "Good: limited excess disclosure through skewed classes"
        if score >= 0.4:
            return "Moderate: skewed classes reveal training records' sensitive values"
        return "Poor: sensitive values of training records are disclosed"
