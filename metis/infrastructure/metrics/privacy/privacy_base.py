"""
Base classes for privacy metrics.

This module provides the unified hierarchy of base classes for all privacy metrics:

    BasePrivacyMetric (abstract)
        │
        ├── DatasetPrivacyMetric (dataset-based metrics)
        │       │
        │       ├── AttributeInferenceMetric (MIA, Inference Attack)
        │       ├── ReidentificationMetric (k-Anonymity, l-Diversity, t-Closeness, Record Linkage)
        │       └── EmpiricalSimilarityMetric (DCR, NNAA)
        │
        └── MechanismPrivacyMetric (mechanism-based metrics)
                └── DifferentialPrivacyMetric (DP verification)

All metrics follow the fit/compute pattern and produce normalized scores in [0, 1]
where 1 = most private (best privacy preservation).
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.base import MetricBase

# =============================================================================
# Base Privacy Metric
# =============================================================================


class BasePrivacyMetric(ABC):
    """
    Abstract base class for all privacy metrics.

    Provides common infrastructure for measuring privacy preservation
    in synthetic data generation.

    Attributes:
        name: Short identifier for the metric
        family: Always "privacy" for privacy metrics
        purpose_tags: set of tags describing metric purpose
        higher_is_better: Always True (1 = most private)
    """

    name: str = "base_privacy"
    family: str = "privacy"
    purpose_tags: set = {"privacy"}
    higher_is_better: bool = True  # 1 = most private

    def __init__(self):
        self._real_data: pd.DataFrame | None = None
        self._synth_data: pd.DataFrame | None = None
        self._context: dict[str, Any] = {}

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> "BasePrivacyMetric":
        """
        Initialize metric with data and context.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            context: Execution context (may contain generative_model, etc.)

        Returns:
            Self for method chaining
        """
        # Fair comparison: use same sample size for both datasets
        n_samples = min(len(real_data), len(synth_data))
        seed = context.get("seed", 42)

        # Sample down to fair comparison size
        if len(real_data) > n_samples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(real_data), size=n_samples, replace=False)
            real_data = real_data.iloc[indices].reset_index(drop=True)

        if len(synth_data) > n_samples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(synth_data), size=n_samples, replace=False)
            synth_data = synth_data.iloc[indices].reset_index(drop=True)

        self._real_data = real_data
        self._synth_data = synth_data
        self._context = context
        self._on_fit()
        return self

    def _on_fit(self) -> None:
        """Hook for subclasses to perform additional setup after fit."""
        return

    @abstractmethod
    def compute(self) -> MetricResult:
        """
        Compute the privacy metric.

        Returns:
            MetricResult with value in [0, 1] where 1 = most private
        """
        pass

    def _get_numeric_columns(self) -> list[str]:
        """Get common numeric columns between real and synthetic data."""
        if self._real_data is None or self._synth_data is None:
            return []
        real_num = set(self._real_data.select_dtypes(include=[np.number]).columns)
        synth_num = set(self._synth_data.select_dtypes(include=[np.number]).columns)
        return sorted(real_num & synth_num)

    def _get_categorical_columns(self) -> list[str]:
        """Get common categorical columns between real and synthetic data."""
        if self._real_data is None or self._synth_data is None:
            return []
        real_cat = set(self._real_data.select_dtypes(include=["object", "category"]).columns)
        synth_cat = set(self._synth_data.select_dtypes(include=["object", "category"]).columns)
        return sorted(real_cat & synth_cat)

    def _create_error_result(self, error_msg: str) -> MetricResult:
        """Create a MetricResult for error cases (NaN so aggregation skips it)."""
        return MetricResult(
            id=f"privacy.{self.name}",
            value=float("nan"),
            details={"error": error_msg},
            family=self.family,
            purpose_tags=self.purpose_tags,
        )


# =============================================================================
# Dataset-based Privacy Metrics
# =============================================================================


class DatasetPrivacyMetric(BasePrivacyMetric, MetricBase):
    """
    Base class for dataset-based privacy metrics.

    These metrics measure privacy by comparing statistical properties
    between real and synthetic datasets, without requiring access to
    the generative model.

    Inherits from MetricBase to access caching utilities like
    _get_knn_distances().

    Subcategories:
        - Attribute Inference: MIA, Inference Attack
        - Reidentification: k-Anonymity, l-Diversity, t-Closeness, Record Linkage
        - Empirical Similarity: DCR, NNAA
    """

    purpose_tags: set = {"privacy", "dataset_based"}

    def __init__(self):
        BasePrivacyMetric.__init__(self)
        MetricBase.__init__(self)

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> "DatasetPrivacyMetric":
        """Initialize with data, setting up both base classes."""
        BasePrivacyMetric.fit(self, real_data, synth_data, context)
        self._setup(real_data, synth_data, context)
        return self

    def _compute_knn_distances(self, k: int = 5) -> dict[str, Any]:
        """
        Compute k-nearest neighbor distances for privacy metrics.

        This is essential for metrics like DCR (Distance to Closest Record)
        and NNAA (Nearest Neighbor Adversarial Accuracy).

        Args:
            k: Number of neighbors to find

        Returns:
            Dictionary with distances and indices, or error info
        """
        numeric_cols = self._get_numeric_columns()
        if not numeric_cols:
            return {"error": "No numeric columns found for KNN computation"}

        try:
            real_numeric = self._real_data[numeric_cols].fillna(0)
            synth_numeric = self._synth_data[numeric_cols].fillna(0)

            n_real = len(real_numeric)
            n_synth = len(synth_numeric)

            if n_real < k or n_synth < 1:
                return {
                    "error": f"Insufficient data points: {n_real} real, {n_synth} synth, need k={k}"
                }

            # Standardize data
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_numeric)
            synth_scaled = scaler.transform(synth_numeric)

            # Fit KNN on real data
            knn_real = NearestNeighbors(n_neighbors=min(k, n_real))
            knn_real.fit(real_scaled)

            # Find distances from synthetic to real
            distances_synth_to_real, indices_synth_to_real = knn_real.kneighbors(synth_scaled)

            # Also fit KNN on synthetic for bidirectional metrics
            knn_synth = NearestNeighbors(n_neighbors=min(k, n_synth))
            knn_synth.fit(synth_scaled)

            # Find distances from real to synthetic
            distances_real_to_synth, indices_real_to_synth = knn_synth.kneighbors(real_scaled)

            return {
                # Synth → Real distances
                "distances_synth_to_real": distances_synth_to_real,
                "indices_synth_to_real": indices_synth_to_real,
                # Real → Synth distances
                "distances_real_to_synth": distances_real_to_synth,
                "indices_real_to_synth": indices_real_to_synth,
                # Stats
                "mean_distance_synth_to_real": float(np.mean(distances_synth_to_real[:, 0])),
                "min_distance_synth_to_real": float(np.min(distances_synth_to_real[:, 0])),
                "max_distance_synth_to_real": float(np.max(distances_synth_to_real[:, 0])),
                "k": k,
                "n_real": n_real,
                "n_synth": n_synth,
            }
        except Exception as e:
            return {"error": f"KNN computation failed: {str(e)}"}


# =============================================================================
# Attribute Inference Metrics
# =============================================================================


class AttributeInferenceMetric(DatasetPrivacyMetric):
    """
    Base class for attribute inference privacy metrics.

    These metrics measure vulnerability to attacks that try to infer
    sensitive attributes from synthetic data.

    Includes:
        - MIA (Membership Inference Attack)
        - Inference Attack (attribute inference)
    """

    purpose_tags: set = {"privacy", "dataset_based", "attribute_inference"}


# =============================================================================
# Reidentification Metrics
# =============================================================================


class ReidentificationMetric(DatasetPrivacyMetric):
    """
    Base class for reidentification privacy metrics.

    These metrics measure vulnerability to attacks that try to
    link synthetic records back to real individuals.

    Includes:
        - k-Anonymity
        - l-Diversity
        - t-Closeness
        - Record Linkage
    """

    purpose_tags: set = {"privacy", "dataset_based", "reidentification"}


# =============================================================================
# Empirical Similarity Metrics
# =============================================================================


class EmpiricalSimilarityMetric(DatasetPrivacyMetric):
    """
    Base class for empirical similarity privacy metrics.

    These metrics measure privacy by computing distances between
    synthetic and real records.

    Includes:
        - DCR (Distance to Closest Record)
        - NNAA (Nearest Neighbor Adversarial Accuracy)
    """

    purpose_tags: set = {"privacy", "dataset_based", "empirical_similarity"}


# =============================================================================
# Mechanism-based Privacy Metrics
# =============================================================================


class MechanismPrivacyMetric(BasePrivacyMetric):
    """
    Base class for mechanism-based privacy metrics.

    These metrics require access to the generative model to verify
    privacy guarantees provided by the generation mechanism.

    The generative model must be passed via context["generative_model"].

    Configuration:
        - epsilon: Privacy parameter (hardcoded to 1.0 for now)

    Includes:
        - Differential Privacy verification
    """

    purpose_tags: set = {"privacy", "mechanism_based"}

    # Hardcoded epsilon for DP verification
    DEFAULT_EPSILON: float = 1.0

    def __init__(self, epsilon: float | None = None):
        """
        Initialize mechanism-based metric.

        Args:
            epsilon: Privacy parameter. If None, uses DEFAULT_EPSILON (1.0)
        """
        super().__init__()
        self._epsilon = epsilon if epsilon is not None else self.DEFAULT_EPSILON
        self._generative_model: Any | None = None

    def _on_fit(self) -> None:
        """Validate and extract generative model from context."""
        self._generative_model = self._context.get("generative_model")

        # If model has epsilon attribute, use it (unless explicitly overridden)
        if self._generative_model is not None:
            model_epsilon = getattr(self._generative_model, "epsilon", None)
            if model_epsilon is not None and self._epsilon == self.DEFAULT_EPSILON:
                self._epsilon = model_epsilon

    def _validate_model(self) -> str | None:
        """
        Validate that generative model is available.

        Returns:
            Error message if validation fails, None otherwise
        """
        if self._generative_model is None:
            return (
                "Generative model not provided. "
                "Mechanism-based metrics require context['generative_model']."
            )
        return None

    @property
    def epsilon(self) -> float:
        """Get the privacy parameter epsilon."""
        return self._epsilon
