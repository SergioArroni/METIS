"""
Record Linkage Privacy Metric.

Measures vulnerability to record linkage attacks where an adversary
attempts to link synthetic records back to real records.

A linkage rate no higher than for unseen real records indicates good privacy.
Score is normalized to [0, 1] where 1 = most private (linkage fails).
"""

import numpy as np
import pandas as pd

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import (
    PARTIAL_MATCH_MAX_RECORDS,
    AttackMetric,
    audit_strategies,
    cap_records,
    partial_match,
    interpret_epsilon,
    relative_closeness,
    signed_excess_risk,
)


@register("privacy.record_linkage")
class RecordLinkageMetric(AttackMetric):
    """
    Record Linkage Privacy Metric.

    Linkage attack on the quasi-identifiers: the adversary knows a person's
    linking columns (every shared column except the target, or
    ``linking_columns``) and looks for a synthetic record that matches them
    unusually well.

    1. For every training record (victim) and every holdout record
       (``context["holdout"]``, the control: people the generator never
       saw), the attack score is how much closer its closest synthetic
       record is than its closest other real record, on the linking columns
       (``relative_closeness``).
    2. The attack is audited at low false-positive rates (``audit_epsilon``):
       ``privacy = e^{−ε̂}``.

    Interpretation:
        - victims link no better than unseen people → ε̂ = 0, score 1
        - some victims link far better (copies, near-copies on the
          quasi-identifiers) → score drops even if they are a few percent

    Until 2026-10-14 the control was the holdout's distances to the *real*
    table and the score an average linkage rate: a generator farther from
    the real rows than the holdout offset its copies (25 % of copies kept
    the score at 1.00 on adult).  ``details`` keeps the average linkage rate
    at ``reference_quantile`` of the control's distances.

    References:
        - Fellegi & Sunter (1969): A Theory for Record Linkage
        - Christen (2012): Data Matching: Concepts and Techniques
        - Giomi et al. (2023): A Unified Framework for Quantifying Privacy Risk
          in Synthetic Data (Anonymeter)
    """

    name: str = "record_linkage"
    purpose_tags: set = {
        "privacy",
        "dataset_based",
        "attacks",
        "record_linkage",
    }

    def __init__(
        self,
        linking_columns: list[str] | None = None,
        reference_quantile: float = 0.5,
        distance_threshold: float | None = None,
    ):
        """
        Initialize Record Linkage metric.

        Args:
            linking_columns: Columns the attacker links on. ``None`` uses every
                shared column except the target (the quasi-identifiers).
            reference_quantile: Quantile of the reference distances used as
                the linkage threshold when ``distance_threshold`` is None.
            distance_threshold: Explicit linkage threshold in the encoded
                space (standardised numerics, one-hot categoricals).
        """
        super().__init__()
        self.linking_columns = linking_columns
        self.reference_quantile = reference_quantile
        self.distance_threshold = distance_threshold

    def compute(self) -> MetricResult:
        """
        Compute Record Linkage privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            holdout = self._get_holdout()
            if holdout is None:
                return self._no_holdout_result("Record linkage")
            linking_cols, dropped = self._resolve_quasi_identifiers(self.linking_columns, [])
            if not linking_cols:
                return self._create_error_result("No linking columns found")
            if len(self._real_data) < 2 or len(self._synth_data) < 1:
                return self._create_error_result(
                    f"Insufficient data: {len(self._real_data)} real, "
                    f"{len(self._synth_data)} synthetic records"
                )

            frames = [self._real_data, holdout]
            d_victims, d_control = self._nn_distance_to_synth(frames, linking_cols)
            local = self._local_scale(frames, linking_cols)
            # Second strategy: the linking values of a victim copied verbatim in one synthetic row.
            rng = np.random.default_rng(self._context.get("seed", 42))
            pm = [cap_records(f, PARTIAL_MATCH_MAX_RECORDS, rng) for f in frames]
            (bits_v, _), (bits_c, _) = partial_match(
                pm, self._synth_data, pd.concat(frames), linking_cols
            )
            audit = audit_strategies(
                [
                    (relative_closeness(d_victims, local[0]), relative_closeness(d_control, local[1])),
                    (bits_v, bits_c),
                ],
                ["whole_record_closeness", "partial_exact_match"],
            )
            privacy_score = audit["privacy_score"]

            if self.distance_threshold is not None:
                threshold = float(self.distance_threshold)
            else:
                threshold = float(np.quantile(d_control, self.reference_quantile))
            linkage_rate = float(np.mean(d_victims <= threshold))
            reference_rate = float(np.mean(d_control <= threshold))

            details = {
                "privacy_score": float(privacy_score),
                "epsilon_lower_bound": audit["epsilon"],
                "audit": audit,
                "excess_risk_signed": signed_excess_risk(linkage_rate, reference_rate),
                "linkage_rate": linkage_rate,
                "reference_linkage_rate": reference_rate,
                "threshold": threshold,
                "fraction_victims_exact_match": float(np.mean(d_victims == 0)),
                "fraction_control_exact_match": float(np.mean(d_control == 0)),
                "linking_columns": linking_cols,
                "excluded_from_linking": dropped,
                "n_real": len(self._real_data),
                "n_holdout": len(holdout),
                "n_synth": len(self._synth_data),
                "interpretation": interpret_epsilon(
                    privacy_score, audit["epsilon"], "training records linkable"
                ),
                **self._holdout_details(),
            }

            return MetricResult(
                id="privacy.record_linkage",
                value=float(privacy_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"Record linkage computation failed: {str(e)}")
