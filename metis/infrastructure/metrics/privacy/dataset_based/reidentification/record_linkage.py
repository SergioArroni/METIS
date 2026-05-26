"""
Record Linkage Privacy Metric.

Measures vulnerability to record linkage attacks where an adversary
attempts to link synthetic records back to real records.

Lower linkage success rate indicates better privacy preservation.
Score is normalized to [0, 1] where 1 = most private (linkage fails).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import ReidentificationMetric


@register("privacy.record_linkage")
class RecordLinkageMetric(ReidentificationMetric):
    """
    Record Linkage Privacy Metric.

    Measures how easily synthetic records can be linked back to
    real records using nearest neighbor matching.

    Process:
    1. For each synthetic record, find nearest real record
    2. Check if the match is "correct" (same quasi-identifier values)
    3. Compute linkage success rate
    4. Convert to privacy score

    A lower linkage rate indicates better privacy - synthetic records
    are not obviously mapped to specific real records.

    Interpretation:
        - Linkage rate ~0%: Cannot link records → Score = 1.0 (good privacy)
        - Linkage rate ~100%: Perfect linkage → Score = 0.0 (poor privacy)

    References:
        - Fellegi & Sunter (1969): A Theory for Record Linkage
        - Christen (2012): Data Matching: Concepts and Techniques
    """

    name: str = "record_linkage"
    purpose_tags: set = {
        "privacy",
        "dataset_based",
        "reidentification",
        "record_linkage",
    }

    def __init__(
        self,
        linking_columns: list[str] | None = None,
        n_neighbors: int = 1,
        distance_threshold: float | None = None,
    ):
        """
        Initialize Record Linkage metric.

        Args:
            linking_columns: Columns to use for linkage. If None, uses all numeric.
            n_neighbors: Number of nearest neighbors to consider
            distance_threshold: Maximum distance to consider a link valid.
                              If None, always links to nearest.
        """
        super().__init__()
        self.linking_columns = linking_columns
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold

    def _get_linking_columns(self) -> list[str]:
        """Get columns to use for linkage."""
        if self.linking_columns:
            all_cols = set(self._real_data.columns) & set(self._synth_data.columns)
            return [c for c in self.linking_columns if c in all_cols]
        return self._get_numeric_columns()

    def _prepare_data(
        self,
        data: pd.DataFrame,
        columns: list[str],
    ) -> np.ndarray:
        """Prepare data for linkage by encoding and scaling."""
        result = data[columns].copy()

        # Encode categorical columns
        for col in columns:
            if (
                result[col].dtype == "object"
                or result[col].dtype.name == "category"
                or pd.api.types.is_string_dtype(result[col])
            ):
                le = LabelEncoder()
                result[col] = le.fit_transform(result[col].astype(str))

        # Fill missing values
        result = result.fillna(0)

        return result.values

    def compute(self) -> MetricResult:
        """
        Compute Record Linkage privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            linking_cols = self._get_linking_columns()
            if not linking_cols:
                return self._create_error_result("No linking columns found")

            # Prepare data
            real_data = self._prepare_data(self._real_data, linking_cols)
            synth_data = self._prepare_data(self._synth_data, linking_cols)

            n_real = len(real_data)
            n_synth = len(synth_data)

            if n_real < self.n_neighbors or n_synth < 1:
                return self._create_error_result(
                    f"Insufficient data: {n_real} real, {n_synth} synthetic records"
                )

            # Scale data
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_data)
            synth_scaled = scaler.transform(synth_data)

            # Find nearest neighbors
            knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
            knn.fit(real_scaled)

            distances, indices = knn.kneighbors(synth_scaled)

            # Compute linkage statistics
            min_distances = distances[:, 0]  # Distance to nearest neighbor

            # If threshold is set, count valid links
            if self.distance_threshold is not None:
                n_linked = np.sum(min_distances <= self.distance_threshold)
                linkage_rate = n_linked / n_synth
            else:
                # Use median distance as implicit threshold
                median_dist = np.median(min_distances)
                n_linked = np.sum(min_distances <= median_dist)
                linkage_rate = n_linked / n_synth

            # Compute uniqueness of linkages
            unique_links = len(set(indices[:, 0]))
            link_uniqueness = unique_links / min(n_real, n_synth)

            # Compute distance statistics
            mean_dist = float(np.mean(min_distances))
            std_dist = float(np.std(min_distances))
            min_dist = float(np.min(min_distances))
            max_dist = float(np.max(min_distances))

            # Privacy score based on distance and linkage rate
            # Higher distances = better privacy
            # Lower linkage rate = better privacy

            # Normalize distance score (assumes distances are in standardized space)
            # Mean distance > 2 (2 std deviations) is considered good
            distance_score = min(1.0, mean_dist / 2.0)

            # Linkage score: 1 - linkage_rate
            linkage_score = 1.0 - linkage_rate

            # Combined score (average)
            privacy_score = (distance_score + linkage_score) / 2.0
            privacy_score = max(0.0, min(1.0, privacy_score))

            details = {
                "linkage_rate": float(linkage_rate),
                "n_linked_records": int(n_linked),
                "unique_links": unique_links,
                "link_uniqueness": float(link_uniqueness),
                "distance_stats": {
                    "mean": mean_dist,
                    "std": std_dist,
                    "min": min_dist,
                    "max": max_dist,
                    "median": float(np.median(min_distances)),
                },
                "distance_threshold": self.distance_threshold,
                "n_neighbors": self.n_neighbors,
                "linking_columns": linking_cols,
                "n_real": n_real,
                "n_synth": n_synth,
                "component_scores": {
                    "distance_score": float(distance_score),
                    "linkage_score": float(linkage_score),
                },
                "interpretation": self._interpret_score(privacy_score, linkage_rate),
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

    def _interpret_score(self, score: float, linkage_rate: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        if score >= 0.8:
            return f"Excellent privacy - linkage rate {linkage_rate:.1%}, records well-separated"
        if score >= 0.6:
            return f"Good privacy - linkage rate {linkage_rate:.1%}, reasonable separation"
        if score >= 0.4:
            return f"Moderate privacy - linkage rate {linkage_rate:.1%}, some records close"
        if score >= 0.2:
            return f"Poor privacy - linkage rate {linkage_rate:.1%}, many records linkable"
        return f"Critical privacy risk - linkage rate {linkage_rate:.1%}, records easily linked"
