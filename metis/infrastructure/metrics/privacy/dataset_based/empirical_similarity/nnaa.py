"""
Nearest Neighbor Adversarial Accuracy (NNAA) Privacy Metric.

Measures privacy by computing how well a classifier can distinguish
between real and synthetic records using nearest neighbor relationships.

Score is normalized to [0, 1] where 1 = most private (classifier fails).
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import EmpiricalSimilarityMetric

# Distances closer than this (relative) count as a tie.
_TIE_TOLERANCE = 1e-9
# Subsample draws averaged by the privacy loss (the three tables are cut to
# the same size in each draw).
_N_DRAWS = 5


@register("privacy.nnaa")
class NNAAMetric(EmpiricalSimilarityMetric):
    """
    Nearest Neighbor Adversarial Accuracy (NNAA) Privacy Metric.

    Privacy loss of Yale et al. (2020): the nearest-neighbour adversarial
    accuracy of the synthetic table against the training records, ``AA_TS``,
    compared with the same accuracy against the holdout, ``AA_ES``
    (``context["holdout"]``):

        AA_XS = ½·[P(d_XS(x) > d_XX(x)) + P(d_SX(s) > d_SS(s))]

    (``d_XS(x)``: distance from a record of X to its closest synthetic
    record; ``d_XX``: to its closest other record of X; ties split).  Each
    term is the accuracy of a 1-NN classifier telling X from S.  A generator
    that memorised its training rows looks *more* like them than like the
    holdout: ``AA_TS`` drops below ``AA_ES``.

        privacy loss = AA_ES − AA_TS,  privacy = 1 − clip(loss / 0.5, 0, 1)

    (0.5 is the loss of an exact copy: ``AA_TS = 0``, ``AA_ES ≈ 0.5``).  The
    three tables are subsampled to the same size, as AA requires.  An
    average-case view, like DCR.  Until 2026-10-14 the score was the
    accuracy against the real table alone: it measured resemblance, so a
    generator farther from the real rows hid its copies (25 % of copies kept
    it at 0.86 on adult) and an independent sample scored 0.91 on telco.

    References:
        - Yale et al. (2019): Privacy Preserving Synthetic Health Data
        - Jordon et al. (2019): PATE-GAN: Generating Synthetic Data with DP
    """

    name: str = "nnaa"
    purpose_tags: set = {
        "privacy",
        "dataset_based",
        "empirical_similarity",
        "nnaa",
        "adversarial",
    }

    def __init__(self, exclude_self: bool = True):
        """
        Initialize NNAA metric.

        Args:
            exclude_self: Must be True: every synthetic query belongs to the
                combined set, so without excluding it each query is its own
                nearest neighbour and the score is always 1. Kept for
                configuration compatibility.
        """
        if not exclude_self:
            raise ValueError(
                "exclude_self=False makes every synthetic record its own nearest "
                "neighbour (NNAA would always be 1); it must be True"
            )
        super().__init__()
        self.exclude_self = exclude_self

    def compute(self) -> MetricResult:
        """
        Compute NNAA privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            holdout = self._get_holdout()
            if holdout is None:
                return self._no_holdout_result("NNAA")
            columns = self._common_columns()
            if not columns:
                return self._create_error_result("No shared columns found")
            n = min(len(self._real_data), len(holdout), len(self._synth_data))
            if n < 2:
                return self._create_error_result(
                    f"Insufficient data: {len(self._real_data)} real, {len(holdout)} holdout, "
                    f"{len(self._synth_data)} synthetic records"
                )

            rng = np.random.default_rng(self._context.get("seed", 42))
            encoder = self._real_distance_space(columns)["encoder"]

            def sample(frame) -> np.ndarray:
                rows = np.sort(rng.choice(len(frame), n, replace=False))
                return encoder.transform(frame.iloc[rows])

            # Averaged over draws: one draw of 100 rows put an independent
            # sample at a loss of 0.17.
            aa_ts, aa_es = [], []
            for _ in range(_N_DRAWS):
                train, hold = sample(self._real_data), sample(holdout)
                synth = sample(self._synth_data)
                aa_ts.append(self._adversarial_accuracy(train, synth))
                aa_es.append(self._adversarial_accuracy(hold, synth))
            aa_ts, aa_es = float(np.mean(aa_ts)), float(np.mean(aa_es))
            loss = aa_es - aa_ts
            privacy_score = 1.0 - min(1.0, max(0.0, loss / 0.5))

            details = {
                "privacy_score": float(privacy_score),
                "privacy_loss": float(loss),
                "excess_risk_signed": float(loss / 0.5),
                "aa_training_synthetic": aa_ts,
                "aa_holdout_synthetic": aa_es,
                "n_per_table": n,
                "columns": columns,
                "interpretation": self._interpret_score(privacy_score, loss),
                **self._holdout_details(),
            }

            return MetricResult(
                id="privacy.nnaa",
                value=float(privacy_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"NNAA computation failed: {str(e)}")

    @classmethod
    def _adversarial_accuracy(cls, x: np.ndarray, synth: np.ndarray) -> float:
        """``AA_XS``: accuracy of the 1-NN classifier telling X from S (ties split)."""
        x_index = NearestNeighbors().fit(x)
        s_index = NearestNeighbors().fit(synth)
        share_x, _ = cls._nn_real_share(x, x_index, s_index, query_is_real=True)
        share_s, _ = cls._nn_real_share(synth, x_index, s_index, query_is_real=False)
        return float(0.5 * (np.mean(share_x) + np.mean(1.0 - share_s)))

    @staticmethod
    def _nn_real_share(
        queries: np.ndarray,
        real_index: NearestNeighbors,
        synth_index: NearestNeighbors,
        *,
        query_is_real: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per query, the probability that its nearest *other* record is real.

        The query itself (it belongs to the real or the synthetic set) is
        excluded by taking the second neighbour in its own set.  Ties at the
        nearest distance are split in proportion to how many real and
        synthetic records share it, i.e. random tie-breaking.  Taking the
        first index instead favoured the real rows (they come first), so with
        discrete data, where exact ties are the rule, an independent sample
        scored as a copy.

        Returns ``(share, nearest_distance)``.
        """
        own, other = (real_index, synth_index) if query_is_real else (synth_index, real_index)
        d_own = own.kneighbors(queries, n_neighbors=2)[0][:, 1]
        d_other = other.kneighbors(queries, n_neighbors=1)[0][:, 0]
        d_real, d_synth = (d_own, d_other) if query_is_real else (d_other, d_own)
        nearest = np.minimum(d_real, d_synth)
        tol = _TIE_TOLERANCE * (1.0 + nearest)
        share = np.where(d_real < d_synth, 1.0, 0.0)
        tied = np.abs(d_real - d_synth) <= tol
        if tied.any():
            # Records within the nearest distance; the query itself is in its own set.
            radius = nearest[tied] + tol[tied]
            q = queries[tied]
            n_real = np.empty(q.shape[0])
            n_synth = np.empty(q.shape[0])
            # One radius query per distinct tie distance (discrete data: very few).
            for r in np.unique(radius):
                rows = radius == r
                n_real[rows] = [len(i) for i in real_index.radius_neighbors(q[rows], r)[1]]
                n_synth[rows] = [len(i) for i in synth_index.radius_neighbors(q[rows], r)[1]]
            if query_is_real:
                n_real = n_real - 1
            else:
                n_synth = n_synth - 1
            share[tied] = n_real / np.maximum(n_real + n_synth, 1)
        return share, nearest

    def _interpret_score(self, score: float, loss: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        msg = f"privacy loss AA_ES − AA_TS = {loss:.3f}"
        if score >= 0.9:
            return f"No average excess - {msg}"
        if score >= 0.7:
            return f"Some excess - {msg}"
        if score >= 0.4:
            return f"Clear excess - {msg}, synthetic records resemble the training rows"
        return f"Severe excess - {msg}, synthetic records are near-copies"
