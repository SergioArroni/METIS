"""
Membership Inference Attack (MIA) Privacy Metric.

Measures vulnerability to membership inference attacks, where an adversary
tries to determine whether a specific record was in the training data.

A lower attack success rate indicates better privacy preservation.
Score is normalized to [0, 1] where 1 = most private (attack fails).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import AttributeInferenceMetric


@register("privacy.mia")
class MembershipInferenceMetric(AttributeInferenceMetric):
    """
    Membership Inference Attack (MIA) Privacy Metric.

    Simulates a membership inference attack using a shadow model approach:
    1. Train a classifier to distinguish between "in" (real) and "out" (synthetic) records
    2. Measure attack accuracy
    3. Convert to privacy score: 1 - (accuracy - 0.5) * 2

    Interpretation:
        - Attack accuracy ~50%: Attacker cannot distinguish → Score ~1.0 (good privacy)
        - Attack accuracy ~100%: Attacker always knows → Score ~0.0 (poor privacy)

    References:
        - Shokri et al. (2017): Membership Inference Attacks Against Machine Learning Models
        - Stadler et al. (2020): Synthetic Data - Anonymisation Groundhog Day
    """

    name: str = "mia"
    purpose_tags: set = {"privacy", "dataset_based", "attribute_inference", "mia"}

    def __init__(self, n_shadow_samples: int = 1000, test_size: float = 0.3):
        """
        Initialize MIA metric.

        Args:
            n_shadow_samples: Number of samples for shadow model training
            test_size: Fraction of data to use for testing
        """
        super().__init__()
        self.n_shadow_samples = n_shadow_samples
        self.test_size = test_size

    def compute(self) -> MetricResult:
        """
        Compute MIA privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            # Get numeric columns
            numeric_cols = self._get_numeric_columns()
            if not numeric_cols:
                return self._create_error_result("No numeric columns found")

            # Prepare data
            real_numeric = self._real_data[numeric_cols].fillna(0).values
            synth_numeric = self._synth_data[numeric_cols].fillna(0).values

            n_real = len(real_numeric)
            n_synth = len(synth_numeric)

            if n_real < 10 or n_synth < 10:
                return self._create_error_result(
                    f"Insufficient data: {n_real} real, {n_synth} synthetic records"
                )

            # Sample data for shadow model
            n_samples = min(self.n_shadow_samples, n_real, n_synth)

            seed = self._context.get("seed", 42)
            rng = np.random.default_rng(seed)
            real_idx = rng.choice(n_real, size=n_samples, replace=n_samples > n_real)
            synth_idx = rng.choice(n_synth, size=n_samples, replace=n_samples > n_synth)

            real_sample = real_numeric[real_idx]
            synth_sample = synth_numeric[synth_idx]

            # Create labels: 1 = real (member), 0 = synthetic (non-member)
            X = np.vstack([real_sample, synth_sample])
            y = np.array([1] * n_samples + [0] * n_samples)

            # Shuffle
            shuffle_idx = rng.permutation(len(X))
            X = X[shuffle_idx]
            y = y[shuffle_idx]

            # Split for training attack model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=seed, stratify=y
            )

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train attack classifier (shadow model)
            import os

            n_jobs = int(os.environ.get("METIS_N_JOBS", "1"))
            attack_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=seed,
                n_jobs=n_jobs,
            )
            attack_model.fit(X_train_scaled, y_train)

            # Evaluate attack
            attack_accuracy = attack_model.score(X_test_scaled, y_test)

            # Get predictions for detailed analysis
            y_pred = attack_model.predict(X_test_scaled)

            # Calculate precision and recall for membership
            tp = np.sum((y_pred == 1) & (y_test == 1))
            fp = np.sum((y_pred == 1) & (y_test == 0))
            fn = np.sum((y_pred == 0) & (y_test == 1))
            tn = np.sum((y_pred == 0) & (y_test == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Convert to privacy score
            # No privacy: accuracy = 0.5 (indistinguishable from real) → score = 0.0
            # Perfect privacy: accuracy = 1.0 (easily distinguished) → score = 1.0
            # Score = 2 * accuracy - 1
            # Clamped to [0, 1]
            privacy_score = max(0.0, min(1.0, 2.0 * attack_accuracy - 1.0))

            # Build details
            details = {
                "attack_accuracy": float(attack_accuracy),
                "attack_precision": float(precision),
                "attack_recall": float(recall),
                "attack_f1": float(f1),
                "privacy_score": float(privacy_score),
                "n_samples_used": n_samples,
                "n_real": n_real,
                "n_synth": n_synth,
                "test_size": self.test_size,
                "confusion_matrix": {
                    "true_positive": int(tp),
                    "false_positive": int(fp),
                    "true_negative": int(tn),
                    "false_negative": int(fn),
                },
                "interpretation": self._interpret_score(privacy_score),
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

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        if score >= 0.9:
            return "Excellent privacy - attack accuracy near random (50%)"
        if score >= 0.7:
            return "Good privacy - attack has limited success"
        if score >= 0.5:
            return "Moderate privacy - some information leakage detected"
        if score >= 0.3:
            return "Poor privacy - significant attack success"
        return "Critical privacy risk - attack highly successful"
