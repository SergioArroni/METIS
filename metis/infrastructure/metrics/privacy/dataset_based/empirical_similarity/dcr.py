"""
Distance to Closest Record (DCR) Privacy Metric.

Measures privacy by computing the distance from each synthetic record
to its nearest real record. Higher distances indicate better privacy.

Score is normalized to [0, 1] where 1 = most private.
"""

import numpy as np

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import EmpiricalSimilarityMetric


@register("privacy.dcr")
class DCRMetric(EmpiricalSimilarityMetric):
    """
    Distance to Closest Record (DCR) Privacy Metric.

    DCR measures the minimum distance from each synthetic record to any
    real record. This directly measures how "close" synthetic data is
    to real data in feature space.

    Low DCR values indicate that synthetic records may be copies or
    near-copies of real records, which is a privacy risk.

    Process:
    1. For each synthetic record, find distance to nearest real record
    2. Compute statistics on these distances
    3. Privacy score based on median/mean DCR

    Interpretation:
        - High DCR: Synthetic records are far from real → Score = 1.0 (good privacy)
        - Low DCR: Synthetic records are close to real → Score = 0.0 (poor privacy)

    References:
        - Zhao et al. (2021): CTAB-GAN: Effective Table Data Synthesizing
        - Synthetic Data Vault metrics
    """

    name: str = "dcr"
    purpose_tags: set = {
        "privacy",
        "dataset_based",
        "empirical_similarity",
        "dcr",
        "distance_based",
    }

    def __init__(
        self,
        k: int = 1,
        percentile_threshold: float = 5.0,
    ):
        """
        Initialize DCR metric.

        Args:
            k: Number of nearest neighbors (typically 1 for DCR)
            percentile_threshold: Percentile below which distances are considered risky
        """
        super().__init__()
        self.k = k
        self.percentile_threshold = percentile_threshold

    def compute(self) -> MetricResult:
        """
        Compute DCR privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            # Compute KNN distances
            knn_results = self._compute_knn_distances(k=max(self.k, 2))

            if "error" in knn_results:
                return self._create_error_result(knn_results["error"])

            # Get distances from synthetic to real (1st nearest neighbor)
            distances = knn_results["distances_synth_to_real"][:, 0]

            n_synth = len(distances)

            # Compute DCR statistics
            dcr_min = float(np.min(distances))
            dcr_max = float(np.max(distances))
            dcr_mean = float(np.mean(distances))
            dcr_median = float(np.median(distances))
            dcr_std = float(np.std(distances))

            # Compute percentiles
            dcr_p5 = float(np.percentile(distances, 5))
            dcr_p10 = float(np.percentile(distances, 10))
            dcr_p25 = float(np.percentile(distances, 25))
            dcr_p75 = float(np.percentile(distances, 75))
            dcr_p90 = float(np.percentile(distances, 90))
            dcr_p95 = float(np.percentile(distances, 95))

            # Count records with very small DCR (potential copies)
            # In standardized space, distance < 0.1 is very close
            copy_threshold = 0.1
            n_potential_copies = int(np.sum(distances < copy_threshold))
            fraction_potential_copies = n_potential_copies / n_synth

            # Privacy score based on 5th percentile (worst case)
            # and fraction of potential copies

            # Score components:
            # 1. Worst case distance (5th percentile)
            # In standardized space, p5 > 1.0 is good (1 std away)
            worst_case_score = min(1.0, dcr_p5 / 1.0) if dcr_p5 > 0 else 0.0

            # 2. Copy fraction penalty
            copy_score = 1.0 - fraction_potential_copies

            # 3. Median distance score
            median_score = min(1.0, dcr_median / 1.5) if dcr_median > 0 else 0.0

            # Combined score (weighted)
            privacy_score = 0.4 * worst_case_score + 0.3 * copy_score + 0.3 * median_score
            privacy_score = max(0.0, min(1.0, privacy_score))

            details = {
                "dcr_stats": {
                    "min": dcr_min,
                    "max": dcr_max,
                    "mean": dcr_mean,
                    "median": dcr_median,
                    "std": dcr_std,
                },
                "dcr_percentiles": {
                    "p5": dcr_p5,
                    "p10": dcr_p10,
                    "p25": dcr_p25,
                    "p75": dcr_p75,
                    "p90": dcr_p90,
                    "p95": dcr_p95,
                },
                "copy_analysis": {
                    "copy_threshold": copy_threshold,
                    "n_potential_copies": n_potential_copies,
                    "fraction_potential_copies": fraction_potential_copies,
                },
                "component_scores": {
                    "worst_case_score": float(worst_case_score),
                    "copy_score": float(copy_score),
                    "median_score": float(median_score),
                },
                "n_synthetic": n_synth,
                "n_real": knn_results["n_real"],
                "k": self.k,
                "interpretation": self._interpret_score(
                    privacy_score, dcr_median, fraction_potential_copies
                ),
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

    def _interpret_score(
        self,
        score: float,
        median_dcr: float,
        copy_fraction: float,
    ) -> str:
        """Provide human-readable interpretation of the privacy score."""
        if score >= 0.8:
            return f"Excellent privacy - median DCR={median_dcr:.3f}, no potential copies"
        if score >= 0.6:
            return f"Good privacy - median DCR={median_dcr:.3f}, {copy_fraction:.1%} close records"
        if score >= 0.4:
            return (
                f"Moderate privacy - median DCR={median_dcr:.3f}, {copy_fraction:.1%} close records"
            )
        if score >= 0.2:
            return (
                f"Poor privacy - median DCR={median_dcr:.3f}, {copy_fraction:.1%} potential copies"
            )
        return "Critical privacy risk - many synthetic records are near-copies of real records"
