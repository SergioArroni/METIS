"""
Membership Inference Attack (MIA) Privacy Metric.

Measures how well an adversary holding the synthetic table can tell whether
a real record was part of the generator's training data.

Score is normalized to [0, 1] where 1 = most private (the attack is no better
than chance).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import (
    PARTIAL_MATCH_MAX_RECORDS,
    AttackMetric,
    audit_strategies,
    cap_records,
    interpret_epsilon,
    partial_match,
    relative_closeness,
)


@register("privacy.mia")
class MembershipInferenceMetric(AttackMetric):
    """
    Membership Inference Attack (MIA) Privacy Metric.

    Distance-based black-box attack on the released synthetic table:

    1. Members are the real records (the generator's training data);
       non-members are the holdout records (``context["holdout"]``), real
       rows from the same population that the generator never saw.
    2. The attack score of a record is how much closer it is to the
       synthetic table than to the other real records (``relative_closeness``:
       distance to the closest synthetic record over distance to the closest
       other real record, members and holdout pooled, in the mixed-type space
       shared with DCR).  Records the generator memorised sit closer to the
       synthetic table than their own neighbourhood explains; dividing by the
       local scale is the distance analogue of DOMIAS' density ratio (van
       Breugel et al. 2023), and catches near-copies of records in dense
       regions that the plain distance misses.
    3. The attack is audited at low false-positive rates, not on average
       (``audit_epsilon``): ``ε̂`` is the lower confidence bound of
       ``log(TPR/FPR)`` over thresholds set on the non-members, and
       ``privacy = e^{−ε̂}``.  The AUC is kept in ``details`` as the
       average-case view.

    Interpretation:
        - members no closer than non-members, even in the tail → ε̂ = 0, score 1
        - some members closer than every non-member (copies, near-copies of
          a few records) → ε̂ > 0; score 0.24 for 5 % of copies in 400 + 400
          records, where the AUC-based score was ≈0.95

    The previous score, ``1 − max(0, 2·AUC − 1)``, averaged over all
    members: a generator that memorised a few records looked private
    (2026-10-14 review of the privacy family, scenarios of copies,
    near-copies and copies of rare records).

    Without a holdout there are no non-members to compare against, so the
    metric returns NaN with an explanatory ``details["error"]`` instead of a
    number (the previous real-vs-synthetic discriminator measured fidelity,
    not membership).

    References:
        - Shokri et al. (2017): Membership Inference Attacks Against Machine Learning Models
        - Carlini et al. (2022): Membership Inference Attacks From First Principles
        - Jagielski et al. (2020): Auditing Differentially Private Machine Learning
        - Stadler et al. (2022): Synthetic Data - Anonymisation Groundhog Day
        - van Breugel et al. (2023): Membership Inference Attacks against Synthetic
          Data through Overfitting Detection (DOMIAS)
    """

    name: str = "mia"
    purpose_tags: set = {"privacy", "dataset_based", "attacks", "mia"}

    def __init__(self, max_records: int = 5000, min_records: int = 10):
        """
        Initialize MIA metric.

        Args:
            max_records: Cap on members and on non-members used (each group
                is subsampled to at most this many records).
            min_records: Minimum members/non-members needed to report a score.
        """
        super().__init__()
        self.max_records = max_records
        self.min_records = min_records

    def compute(self) -> MetricResult:
        """
        Compute MIA privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            holdout = self._get_holdout()
            if holdout is None:
                return self._no_holdout_result("MIA")
            columns = self._common_columns()
            if not columns:
                return self._create_error_result("No shared columns")
            if (
                min(len(self._real_data), len(holdout)) < self.min_records
                or len(self._synth_data) < 1
            ):
                return self._create_error_result(
                    f"Insufficient data: {len(self._real_data)} members, "
                    f"{len(holdout)} non-members, {len(self._synth_data)} synthetic"
                )

            rng = np.random.default_rng(self._context.get("seed", 42))

            def cap(frame):
                if len(frame) <= self.max_records:
                    return frame
                return frame.iloc[np.sort(rng.choice(len(frame), self.max_records, replace=False))]

            members, non_members = cap(self._real_data), cap(holdout)
            d_members, d_non_members = self._nn_distance_to_synth([members, non_members], columns)
            local = self._local_scale([members, non_members], columns)
            # Second strategy: identifying values of a member copied verbatim in one synthetic row.
            pm_m, pm_h = (cap_records(f, PARTIAL_MATCH_MAX_RECORDS, rng) for f in (members, non_members))
            (bits_m, _), (bits_h, _) = partial_match(
                [pm_m, pm_h], self._synth_data, pd.concat([self._real_data, holdout]), columns
            )
            audit = audit_strategies(
                [
                    (relative_closeness(d_members, local[0]), relative_closeness(d_non_members, local[1])),
                    (bits_m, bits_h),
                ],
                ["whole_record_closeness", "partial_exact_match"],
            )
            privacy_score = audit["privacy_score"]

            labels = np.r_[np.ones(len(d_members)), np.zeros(len(d_non_members))]
            auc = float(roc_auc_score(labels, -np.r_[d_members, d_non_members]))

            details = {
                "epsilon_lower_bound": audit["epsilon"],
                "privacy_score": privacy_score,
                "audit": audit,
                "attack_auc": auc,
                "membership_advantage": audit["advantage"],
                "excess_risk_signed": float(2.0 * auc - 1.0),
                "median_distance_members": float(np.median(d_members)),
                "median_distance_non_members": float(np.median(d_non_members)),
                "fraction_members_exact_match": float(np.mean(d_members == 0)),
                "fraction_non_members_exact_match": float(np.mean(d_non_members == 0)),
                "n_members": len(d_members),
                "n_non_members": len(d_non_members),
                "n_synth": len(self._synth_data),
                "columns": columns,
                "interpretation": self._interpret_score(privacy_score, audit["epsilon"]),
                **self._holdout_details(),
            }

            return MetricResult(
                id="privacy.mia",
                value=float(privacy_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"MIA computation failed: {str(e)}")

    def _interpret_score(self, score: float, epsilon: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        return interpret_epsilon(score, epsilon, "training records closer to the synthetic table")
