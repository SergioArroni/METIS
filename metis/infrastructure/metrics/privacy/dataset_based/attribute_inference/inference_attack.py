"""
Inference Attack Privacy Metric.

Measures vulnerability to attribute inference attacks, where an adversary
tries to infer sensitive attributes from other known attributes.

A lower inference success rate indicates better privacy preservation.
Score is normalized to [0, 1] where 1 = most private (inference fails).
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.registry import register

from ...privacy_base import AttributeInferenceMetric


@register("privacy.inference_attack")
class InferenceAttackMetric(AttributeInferenceMetric):
    """
    Attribute Inference Attack Privacy Metric.

    Measures how well an attacker can infer sensitive attributes from
    synthetic data, given knowledge of other attributes.

    Process:
    1. For each sensitive column, train a model on synthetic data to predict it
    2. Test on real data to see if inference succeeds
    3. Compare against random baseline
    4. Privacy score reflects how much better than random the attack is

    Interpretation:
        - Inference accuracy = random: Score ~1.0 (good privacy)
        - Inference accuracy = perfect: Score ~0.0 (poor privacy)

    References:
        - Yeom et al. (2018): Privacy Risk in Machine Learning
        - Stadler et al. (2020): Synthetic Data - Anonymisation Groundhog Day
    """

    name: str = "inference_attack"
    purpose_tags: set = {"privacy", "dataset_based", "attribute_inference", "inference"}

    def __init__(
        self,
        sensitive_columns: list[str] | None = None,
        cv_folds: int = 3,
    ):
        """
        Initialize Inference Attack metric.

        Args:
            sensitive_columns: Columns to test for inference. If None, tests all columns.
            cv_folds: Number of cross-validation folds
        """
        super().__init__()
        self.sensitive_columns = sensitive_columns
        self.cv_folds = cv_folds

    def _get_target_columns(self) -> list[str]:
        """Get columns to test for inference attacks."""
        if self.sensitive_columns:
            # Filter to columns that exist in both datasets
            all_cols = set(self._real_data.columns) & set(self._synth_data.columns)
            return [c for c in self.sensitive_columns if c in all_cols]
        # Use all common columns
        return list(set(self._real_data.columns) & set(self._synth_data.columns))

    def _compute_inference_score(
        self,
        target_col: str,
        feature_cols: list[str],
    ) -> dict[str, Any]:
        """
        Compute inference attack score for a single target column.

        Returns:
            Dictionary with attack metrics and privacy score
        """
        try:
            # Prepare features
            X_synth = self._synth_data[feature_cols].copy()
            X_real = self._real_data[feature_cols].copy()

            y_synth = self._synth_data[target_col].copy()
            y_real = self._real_data[target_col].copy()

            # Handle missing values
            X_synth = X_synth.fillna(0)
            X_real = X_real.fillna(0)

            # Encode categorical features
            for col in feature_cols:
                if (
                    X_synth[col].dtype == "object"
                    or X_synth[col].dtype.name == "category"
                    or pd.api.types.is_string_dtype(X_synth[col])
                ):
                    le = LabelEncoder()
                    combined = pd.concat([X_synth[col], X_real[col]]).astype(str)
                    le.fit(combined)
                    X_synth[col] = le.transform(X_synth[col].astype(str))
                    X_real[col] = le.transform(X_real[col].astype(str))

            X_synth = X_synth.values
            X_real = X_real.values

            # Scale features
            scaler = StandardScaler()
            X_synth_scaled = scaler.fit_transform(X_synth)
            X_real_scaled = scaler.transform(X_real)

            # Determine if classification or regression
            is_categorical = (
                y_synth.dtype == "object"
                or y_synth.dtype.name == "category"
                or pd.api.types.is_string_dtype(y_synth)
                or y_synth.nunique() <= 20
            )

            if is_categorical:
                return self._classification_inference(
                    X_synth_scaled, X_real_scaled, y_synth, y_real, target_col
                )
            return self._regression_inference(
                X_synth_scaled, X_real_scaled, y_synth, y_real, target_col
            )

        except Exception as e:
            return {
                "column": target_col,
                "error": str(e),
                "privacy_score": 1.0,  # Assume privacy preserved on error
            }

    def _classification_inference(
        self,
        X_synth: np.ndarray,
        X_real: np.ndarray,
        y_synth: pd.Series,
        y_real: pd.Series,
        target_col: str,
    ) -> dict[str, Any]:
        """Perform classification-based inference attack."""
        # Encode target
        le = LabelEncoder()
        combined_y = pd.concat([y_synth, y_real]).astype(str)
        le.fit(combined_y)
        y_synth_enc = le.transform(y_synth.astype(str))
        y_real_enc = le.transform(y_real.astype(str))

        n_classes = len(le.classes_)
        random_baseline = 1.0 / n_classes

        # Train on synthetic, test on real
        seed = self._context.get("seed", 42)
        import os

        n_jobs = int(os.environ.get("METIS_N_JOBS", "1"))
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
            n_jobs=n_jobs,
        )
        model.fit(X_synth, y_synth_enc)

        # Inference accuracy on real data
        inference_accuracy = model.score(X_real, y_real_enc)

        # Privacy score: how much better than random?
        # If accuracy = random → score = 1.0
        # If accuracy = 1.0 → score = 0.0
        # Normalize: (accuracy - random) / (1 - random) gives advantage
        # Privacy = 1 - advantage
        advantage = max(0, (inference_accuracy - random_baseline) / (1 - random_baseline))
        privacy_score = max(0.0, min(1.0, 1.0 - advantage))

        return {
            "column": target_col,
            "task_type": "classification",
            "n_classes": n_classes,
            "random_baseline": float(random_baseline),
            "inference_accuracy": float(inference_accuracy),
            "advantage": float(advantage),
            "privacy_score": float(privacy_score),
        }

    def _regression_inference(
        self,
        X_synth: np.ndarray,
        X_real: np.ndarray,
        y_synth: pd.Series,
        y_real: pd.Series,
        target_col: str,
    ) -> dict[str, Any]:
        """Perform regression-based inference attack."""
        y_synth_vals = y_synth.fillna(0).values
        y_real_vals = y_real.fillna(0).values

        # Train on synthetic
        seed = self._context.get("seed", 42)
        import os

        n_jobs = int(os.environ.get("METIS_N_JOBS", "1"))
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
            n_jobs=n_jobs,
        )
        model.fit(X_synth, y_synth_vals)

        # Predict on real
        y_pred = model.predict(X_real)

        # Compute R² score
        ss_res = np.sum((y_real_vals - y_pred) ** 2)
        ss_tot = np.sum((y_real_vals - np.mean(y_real_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2 = max(0, r2)  # Clamp negative R²

        # Privacy score: R² = 0 (random) → 1.0, R² = 1 (perfect) → 0.0
        privacy_score = max(0.0, min(1.0, 1.0 - r2))

        # Also compute normalized RMSE
        rmse = np.sqrt(np.mean((y_real_vals - y_pred) ** 2))
        std_real = np.std(y_real_vals)
        nrmse = rmse / std_real if std_real > 0 else 0.0

        return {
            "column": target_col,
            "task_type": "regression",
            "r2_score": float(r2),
            "rmse": float(rmse),
            "nrmse": float(nrmse),
            "privacy_score": float(privacy_score),
        }

    def compute(self) -> MetricResult:
        """
        Compute Inference Attack privacy metric.

        Returns:
            MetricResult with privacy score in [0, 1] where 1 = most private
        """
        try:
            target_columns = self._get_target_columns()
            if not target_columns:
                return self._create_error_result("No columns available for inference attack")

            # Compute inference score for each target column
            column_results = {}
            privacy_scores = []

            for target_col in target_columns:
                # Use all other columns as features
                feature_cols = [c for c in target_columns if c != target_col]

                if not feature_cols:
                    continue

                result = self._compute_inference_score(target_col, feature_cols)
                column_results[target_col] = result

                if "error" not in result:
                    privacy_scores.append(result["privacy_score"])

            if not privacy_scores:
                return self._create_error_result(
                    "Could not compute inference attack for any column"
                )

            # Aggregate privacy scores (use minimum - worst case)
            overall_score = float(np.mean(privacy_scores))
            min_score = float(np.min(privacy_scores))

            # Find most vulnerable column
            most_vulnerable = min(
                column_results.items(), key=lambda x: x[1].get("privacy_score", 1.0)
            )

            details = {
                "overall_privacy_score": overall_score,
                "min_privacy_score": min_score,
                "n_columns_tested": len(privacy_scores),
                "most_vulnerable_column": most_vulnerable[0],
                "most_vulnerable_score": most_vulnerable[1].get("privacy_score", 1.0),
                "column_results": column_results,
                "interpretation": self._interpret_score(overall_score),
            }

            return MetricResult(
                id="privacy.inference_attack",
                value=float(overall_score),
                details=details,
                family=self.family,
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._create_error_result(f"Inference attack computation failed: {str(e)}")

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of the privacy score."""
        if score >= 0.9:
            return "Excellent privacy - attribute inference near random baseline"
        if score >= 0.7:
            return "Good privacy - limited attribute inference success"
        if score >= 0.5:
            return "Moderate privacy - some attributes can be inferred"
        if score >= 0.3:
            return "Poor privacy - significant attribute inference possible"
        return "Critical privacy risk - attributes highly inferable"
