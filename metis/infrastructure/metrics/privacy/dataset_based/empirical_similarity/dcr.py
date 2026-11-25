"""
Distance to Closest Record (DCR) Privacy Metric.

Compares the distance from each synthetic record to its closest real record
against the same distance for records the generator never saw.

Score is normalized to [0, 1] where 1 = most private.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import EmpiricalSimilarityMetric, excess_risk_score, signed_excess_risk


@register("privacy.dcr")
class DCRMetric(EmpiricalSimilarityMetric):
    """
    Distance to Closest Record (DCR) Privacy Metric.

    Holdout-based DCR test (Platzer & Reutterer 2021; SDMetrics'
    ``DCROverfittingProtection``), the average-case view of the family:

    1. For each synthetic record, is its closest real record a training
       record or a holdout record (``context["holdout"]``)?  Mixed-type space
       (standardised numerics, weighted one-hot categoricals, missingness
       indicators); ties split in proportion.
    2. Without memorisation the synthetic table is no closer to the training
       rows than to the holdout: the share is ``b = n_train / (n_train +
       n_holdout)`` (both pooled, so unequal sizes need no subsampling).
    3. ``risk = (share − b) / (1 − b)`` clipped to [0, 1]; ``privacy = 1 − risk``.

    The share of copies is read directly (5 % of copies → ≈0.95): it is an
    average and does not single out the copied records; MIA and record
    linkage audit the tail.  Until 2026-10-14 the score compared the
    synthetic→real distances with the holdout→real ones (another release):
    a generator farther from the real rows than the holdout offset its
    copies (25 % of copies scored 1.00 on adult).  Average DCR tests are
    criticised as privacy evidence (Yao et al. 2025, "The DCR Delusion"): a
    score of 1 here is necessary, not sufficient.

    References:
        - Zhao et al. (2021): CTAB-GAN: Effective Table Data Synthesizing
        - SDMetrics: DCROverfittingProtection; Synthcity: DistanceToClosestRecord
    """

    name: str = "dcr"
    purpose_tags: set = {
        "privacy",
        "dataset_based",
        "empirical_similarity",
        "dcr",
        "distance_based",
    }

    def __init__(self, columns: list[str] | None = None):
        """
        Initialize DCR metric.

        Args:
            columns: Columns that define a record. ``None`` uses every column
                shared by the real and synthetic tables.
        """
        super().__init__()
        self.columns = columns

    def compute(self) -> MetricResult:
        """
        Compute DCR privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            holdout = self._get_holdout()
            if holdout is None:
                return self._no_holdout_result("DCR")
            common = self._common_columns()
            columns = [c for c in self.columns if c in common] if self.columns else common
            if not columns:
                return self._create_error_result("No shared columns for DCR")
            if len(self._real_data) < 2 or len(self._synth_data) < 1:
                return self._create_error_result(
                    f"Insufficient data: {len(self._real_data)} real, "
                    f"{len(self._synth_data)} synthetic records"
                )

            share, d_train, d_hold = self._share_closest_is_training(holdout, columns)
            n_t, n_h = len(self._real_data), len(holdout)
            baseline = n_t / (n_t + n_h)
            privacy_score, risk = excess_risk_score(share, baseline)

            details = {
                "privacy_score": float(privacy_score),
                "risk": float(risk),
                "share_closest_is_training": share,
                "share_expected": baseline,
                "excess_risk_signed": signed_excess_risk(share, baseline),
                "dcr_to_training": self._summary(d_train),
                "dcr_to_holdout": self._summary(d_hold),
                "fraction_exact_copies": float(np.mean(d_train == 0)),
                "fraction_exact_matches_holdout": float(np.mean(d_hold == 0)),
                "n_synthetic": len(self._synth_data),
                "n_real": n_t,
                "n_holdout": n_h,
                "columns": columns,
                "interpretation": self._interpret_score(privacy_score, share, baseline),
                **self._holdout_details(),
            }

            return MetricResult(
                id="privacy.dcr",
                value=float(privacy_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"DCR computation failed: {str(e)}")

    def _share_closest_is_training(
        self, holdout, columns: list[str]
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Share of synthetic records whose closest real record (training and
        holdout pooled) is a training record; and the distances to the
        closest training / holdout record.

        A synthetic record only says something about membership when it sits
        within the neighbourhood of its closest real record (no farther than
        that record's own closest other real record).  Farther records, and
        ties, count as the expected share ``n_train / (n_train + n_holdout)``:
        otherwise every record of a table far from the real rows picks the
        same few extreme real records, and whether they are training or
        holdout rows is a coin toss that swung the share (0.64 for a table
        3 sd away with no copy at all).
        """
        encoder = self._real_distance_space(columns)["encoder"]
        synth = encoder.transform(self._synth_data)
        frames = [self._real_data, holdout]
        local_t, local_h = self._local_scale(frames, columns)

        def nearest(table) -> tuple[np.ndarray, np.ndarray]:
            index = NearestNeighbors(n_neighbors=1).fit(encoder.transform(table))
            dist, idx = index.kneighbors(synth)
            return np.round(dist[:, 0], 9), idx[:, 0]

        (d_train, i_train), (d_hold, i_hold) = nearest(frames[0]), nearest(frames[1])
        n_t, n_h = len(self._real_data), len(holdout)
        expected = n_t / (n_t + n_h)
        closer = np.where(d_train < d_hold, 1.0, 0.0)
        reach = np.where(d_train < d_hold, local_t[i_train], local_h[i_hold])
        neutral = (d_train == d_hold) | (np.minimum(d_train, d_hold) > reach)
        closer[neutral] = expected
        return float(np.mean(closer)), d_train, d_hold

    @staticmethod
    def _summary(distances: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "p5": float(np.percentile(distances, 5)),
            "p95": float(np.percentile(distances, 95)),
            "min": float(np.min(distances)),
        }

    def _interpret_score(self, score: float, share: float, expected: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        msg = f"{share:.1%} of synthetic records closest to a training record (expected {expected:.1%})"
        if score >= 0.9:
            return f"No average excess - {msg}"
        if score >= 0.7:
            return f"Some excess - {msg}"
        if score >= 0.4:
            return f"Clear excess - {msg}, many records closer to the training rows"
        return f"Severe excess - {msg}, synthetic records are near-copies"
