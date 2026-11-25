"""
Inference Attack Privacy Metric.

Measures how much the synthetic table helps an adversary infer a sensitive
attribute of the generator's training records from their other attributes,
beyond what it allows for records the generator never saw.

Score is normalized to [0, 1] where 1 = most private.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import (
    AttackMetric,
    PARTIAL_MATCH_MAX_RECORDS,
    audit_epsilon,
    audit_strategies,
    cap_records,
    excess_risk_score,
    partial_match,
    relative_closeness,
    signed_excess_risk,
)

# Attack score of a wrong guess: below every right one, so it never counts.
_WRONG_GUESS = -1e9


@register("privacy.inference_attack")
class InferenceAttackMetric(AttackMetric):
    """
    Attribute Inference Attack Privacy Metric.

    The attack of Anonymeter (Giomi et al. 2023): the adversary knows a
    person's other columns, finds the closest synthetic record on them and
    guesses its sensitive value.

    1. Victims: the training records; control: the holdout records
       (``context["holdout"]``), people the generator never saw.  The
       control separates privacy leakage from the generalisable signal a
       useful generator *should* reproduce.
    2. A guess counts when it is right; its confidence is how much closer
       the synthetic match is than the person's closest other real record
       (``relative_closeness``).
    3. Audited at low false-positive rates (``audit_epsilon``): a few
       training records whose sensitive value is disclosed by a near-exact
       synthetic match lower the score even when the average accuracy
       barely moves.  ``privacy = e^{−ε̂}``, worst sensitive column.

    Until 2026-10-14 the attacker was a random forest trained on the
    synthetic table and the score the excess accuracy averaged over all
    victims (the average stays in ``details``); without a holdout it fell
    back to the majority class, which charged utility as risk.  Now the
    metric needs the holdout.  ``n_estimators``/``max_depth`` are accepted
    for configuration compatibility and unused.

    Sensitive columns: ``sensitive_columns`` or, by default, the target
    (``context["target"]``); without either the metric returns NaN instead of
    attacking every column. Numeric sensitive columns with many distinct
    values are generalised into ``n_bins`` quantile bins and attacked as a
    classification (the former train-on-synthetic / test-on-real R² was a
    utility score, not a privacy one).

    References:
        - Yeom et al. (2018): Privacy Risk in Machine Learning
        - Stadler et al. (2022): Synthetic Data - Anonymisation Groundhog Day
        - Giomi et al. (2023): A Unified Framework for Quantifying Privacy Risk
          in Synthetic Data (Anonymeter)
    """

    name: str = "inference_attack"
    purpose_tags: set = {"privacy", "dataset_based", "attacks", "inference"}

    def __init__(
        self,
        sensitive_columns: list[str] | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        n_bins: int = 5,
    ):
        """
        Initialize Inference Attack metric.

        Args:
            sensitive_columns: Columns to attack. If None, the context target.
            n_estimators: Trees in the attacker's random forest.
            max_depth: Depth of the attacker's trees (None = grown fully, as an
                attacker trying to exploit memorised records would).
            n_bins: Quantile bins for numeric sensitive columns.
        """
        super().__init__()
        self.sensitive_columns = sensitive_columns
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_bins = n_bins

    def _attack_column(self, sens_col: str, features: list[str]) -> dict[str, Any]:
        holdout = self._get_holdout()
        frames = [self._real_data, holdout]
        labels = [
            self._generalise(f, [sens_col], self.n_bins)[sens_col].to_numpy()
            for f in (self._synth_data, *frames)
        ]
        encoder = self._real_distance_space(features)["encoder"]
        index = NearestNeighbors(n_neighbors=1).fit(encoder.transform(self._synth_data))
        local = self._local_scale(frames, features)
        scores, correct = [], []
        for frame, truth, scale in zip(frames, labels[1:], local, strict=True):
            dist, idx = index.kneighbors(encoder.transform(frame))
            right = labels[0][idx[:, 0]] == truth
            closeness = relative_closeness(np.round(dist[:, 0], 9), scale)
            scores.append(np.where(right, closeness, _WRONG_GUESS))
            correct.append(right)
        # Second strategy: guess the sensitive value from the synthetic row that reproduces most
        # of the record's other values verbatim.
        rng = np.random.default_rng(self._context.get("seed", 42))
        pm_frames = [cap_records(f, PARTIAL_MATCH_MAX_RECORDS, rng) for f in frames]
        pm_truth = [
            self._generalise(f, [sens_col], self.n_bins)[sens_col].to_numpy() for f in pm_frames
        ]
        pm_scores = []
        for (bits, best), truth in zip(
            partial_match(pm_frames, self._synth_data, pd.concat(frames), features),
            pm_truth,
            strict=True,
        ):
            pm_scores.append(np.where(labels[0][best] == truth, bits, _WRONG_GUESS))
        audit = audit_strategies(
            [(scores[0], scores[1]), (pm_scores[0], pm_scores[1])],
            ["whole_record_closeness", "partial_exact_match"],
        )
        acc_real, baseline = float(np.mean(correct[0])), float(np.mean(correct[1]))
        _, risk = excess_risk_score(acc_real, baseline)
        return {
            "column": sens_col,
            "privacy_score": audit["privacy_score"],
            "epsilon_lower_bound": audit["epsilon"],
            "audit": audit,
            "accuracy_on_training_records": acc_real,
            "baseline_accuracy": baseline,
            "n_classes": int(len(np.unique(labels[1]))),
            "risk": risk,
            "excess_risk_signed": signed_excess_risk(acc_real, baseline),
        }

    def compute(self) -> MetricResult:
        """
        Compute Inference Attack privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            if self._get_holdout() is None:
                return self._no_holdout_result("Inference attack")
            sensitive = self._resolve_sensitive_columns(self.sensitive_columns)
            if self._get_holdout() is None:  # it lacked the sensitive column(s)
                return self._no_holdout_result("Inference attack")
            if not sensitive:
                return self._create_not_applicable_result(
                    "No sensitive column: set sensitive_columns or data.target"
                )
            common = self._common_columns()

            column_results = {}
            for sens_col in sensitive:
                features = [c for c in common if c != sens_col]
                if not features:
                    column_results[sens_col] = {"error": "no feature columns"}
                    continue
                try:
                    column_results[sens_col] = self._attack_column(sens_col, features)
                except Exception as exc:  # one failing column must not hide the rest
                    column_results[sens_col] = {"error": str(exc)}

            scores = [r["privacy_score"] for r in column_results.values() if "error" not in r]
            if not scores:
                return self._create_error_result(
                    f"Could not compute inference attack for any column: {column_results}"
                )

            # Worst column: a leak through one sensitive column is a leak.
            overall_score = float(np.min(scores))
            details = {
                "overall_privacy_score": overall_score,
                "excess_risk_signed": float(
                    np.mean(
                        [
                            r["excess_risk_signed"]
                            for r in column_results.values()
                            if "error" not in r
                        ]
                    )
                ),
                "min_privacy_score": float(np.min(scores)),
                "sensitive_columns": sensitive,
                "n_columns_tested": len(scores),
                "column_results": column_results,
                "interpretation": self._interpret_score(overall_score),
                **self._holdout_details(),
            }

            return MetricResult(
                id="privacy.inference_attack",
                value=overall_score,
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"Inference attack computation failed: {str(e)}")

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        if score >= 0.9:
            return "Excellent privacy - inference on training records no better than baseline"
        if score >= 0.7:
            return "Good privacy - limited excess inference on training records"
        if score >= 0.5:
            return "Moderate privacy - training records' sensitive values partly inferable"
        if score >= 0.3:
            return "Poor privacy - significant excess inference on training records"
        return "Critical privacy risk - sensitive values of training records highly inferable"
