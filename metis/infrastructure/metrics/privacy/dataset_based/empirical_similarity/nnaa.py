"""
Nearest Neighbor Adversarial Accuracy (NNAA) Privacy Metric.

Measures privacy by computing how well a classifier can distinguish
between real and synthetic records using nearest neighbor relationships.

Score is normalized to [0, 1] where 1 = most private (classifier fails).
"""

import numpy as np

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import EmpiricalSimilarityMetric


@register("privacy.nnaa")
class NNAAMetric(EmpiricalSimilarityMetric):
    """
    Nearest Neighbor Adversarial Accuracy (NNAA) Privacy Metric.

    NNAA measures privacy by checking if an adversary can distinguish
    real from synthetic records using nearest neighbor relationships.

    For each synthetic record:
    - Find its nearest neighbor in the combined (real + synthetic) dataset
    - If nearest neighbor is real: potential privacy leak
    - If nearest neighbor is synthetic: good privacy

    The adversarial accuracy is the fraction of synthetic records whose
    nearest neighbor is a real record.

    Process:
    1. Combine real and synthetic datasets
    2. For each synthetic record, find nearest neighbor in combined set
    3. Compute fraction whose nearest neighbor is real
    4. Privacy score = 1 - adversarial accuracy

    Interpretation:
        - NNAA = 0.5: Random (can't distinguish) → Score = 1.0 (good privacy)
        - NNAA = 1.0: All synthetic nearest to real → Score = 0.0 (poor privacy)

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
            exclude_self: If True, excludes the query point itself when searching
                         (important when query is in the search set)
        """
        super().__init__()
        self.exclude_self = exclude_self

    def compute(self) -> MetricResult:
        """
        Compute NNAA privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            # Get numeric columns
            numeric_cols = self._get_numeric_columns()
            if not numeric_cols:
                return self._create_error_result("No numeric columns found")

            # Prepare data
            real_data = self._real_data[numeric_cols].fillna(0).values
            synth_data = self._synth_data[numeric_cols].fillna(0).values

            n_real = len(real_data)
            n_synth = len(synth_data)

            if n_real < 2 or n_synth < 2:
                return self._create_error_result(
                    f"Insufficient data: {n_real} real, {n_synth} synthetic records"
                )

            # Standardize
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_data)
            synth_scaled = scaler.transform(synth_data)

            # Combine datasets
            combined_data = np.vstack([real_scaled, synth_scaled])

            # Build KNN on combined data
            from sklearn.neighbors import NearestNeighbors

            k = 2 if self.exclude_self else 1  # Extra neighbor to exclude self
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(combined_data)

            # Query with synthetic records
            distances, indices = knn.kneighbors(synth_scaled)

            # For each synthetic record, check if NN is real or synthetic
            nn_is_real = []
            nn_distances = []

            for i, (dist, idx) in enumerate(zip(distances, indices)):
                # Get the neighbor index (exclude self if needed)
                if self.exclude_self:
                    # Skip if first neighbor is self
                    synth_idx_in_combined = n_real + i
                    if idx[0] == synth_idx_in_combined:
                        nn_idx = idx[1]
                        nn_dist = dist[1]
                    else:
                        nn_idx = idx[0]
                        nn_dist = dist[0]
                else:
                    nn_idx = idx[0]
                    nn_dist = dist[0]

                # Check if nearest neighbor is real (index < n_real)
                is_real = nn_idx < n_real
                nn_is_real.append(is_real)
                nn_distances.append(nn_dist)

            nn_is_real = np.array(nn_is_real)
            nn_distances = np.array(nn_distances)

            # Compute NNAA (fraction of synthetic whose NN is real)
            adversarial_accuracy = float(np.mean(nn_is_real))

            # Also compute the reverse: for real records, is NN synthetic?
            real_distances, real_indices = knn.kneighbors(real_scaled)

            real_nn_is_synth = []
            for i, (_dist, idx) in enumerate(zip(real_distances, real_indices)):
                if self.exclude_self:
                    if idx[0] == i:  # Self
                        nn_idx = idx[1]
                    else:
                        nn_idx = idx[0]
                else:
                    nn_idx = idx[0]

                is_synth = nn_idx >= n_real
                real_nn_is_synth.append(is_synth)

            real_nn_is_synth = np.array(real_nn_is_synth)
            reverse_accuracy = float(np.mean(real_nn_is_synth))

            # Privacy score: perfect privacy when NNAA = 0.5 (random)
            # NNAA > 0.5: synthetic records cluster near real → poor privacy
            # NNAA < 0.5: synthetic records cluster together → good (but might indicate mode collapse)

            # Score formula: distance from 0.5, penalize high NNAA more
            if adversarial_accuracy >= 0.5:
                # High NNAA: synthetic near real → poor privacy
                privacy_score = 1.0 - 2 * (adversarial_accuracy - 0.5)
            else:
                # Low NNAA: synthetic cluster together → good privacy but flag separately
                privacy_score = 1.0

            privacy_score = max(0.0, min(1.0, privacy_score))

            # Compute distance statistics for additional insight
            mean_nn_distance = float(np.mean(nn_distances))
            median_nn_distance = float(np.median(nn_distances))

            details = {
                "adversarial_accuracy": adversarial_accuracy,
                "reverse_accuracy": reverse_accuracy,  # Real records with synth NN
                "privacy_score": float(privacy_score),
                "synth_to_real_count": int(np.sum(nn_is_real)),
                "synth_to_synth_count": int(np.sum(~nn_is_real)),
                "distance_stats": {
                    "mean": mean_nn_distance,
                    "median": median_nn_distance,
                    "min": float(np.min(nn_distances)),
                    "max": float(np.max(nn_distances)),
                },
                "n_real": n_real,
                "n_synth": n_synth,
                "interpretation": self._interpret_score(privacy_score, adversarial_accuracy),
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

    def _interpret_score(self, score: float, nnaa: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        if score >= 0.9:
            return f"Excellent privacy - NNAA={nnaa:.2f}, near random (0.5)"
        if score >= 0.7:
            return f"Good privacy - NNAA={nnaa:.2f}, limited distinguishability"
        if score >= 0.5:
            return f"Moderate privacy - NNAA={nnaa:.2f}, some synthetic-real clustering"
        if score >= 0.3:
            return f"Poor privacy - NNAA={nnaa:.2f}, synthetic records near real"
        return f"Critical privacy risk - NNAA={nnaa:.2f}, synthetic records very close to real"
