"""
Shared mixin for ML efficiency metrics.

Provides canonical implementations of helper methods that were previously
duplicated across BaseStandaloneStrategyMetric, MLEfficiencyMetric,
ClassificationEfficiencyMetric, and RegressionEfficiencyMetric.

Classes that mix this in are expected to also inherit from MetricBase
(which provides ``_context``, ``_real_data``, ``_synth_data``) and to
set ``self.n_trials`` / ``self.n_runs`` before calling ``_get_trainer``.
"""

from typing import Literal

import pandas as pd

from metis.infrastructure.metrics.utility.ml_efficiency.catboost_trainer import (
    ClassificationTrainer,
    MultiTargetClassificationTrainer,
    MultiTargetRegressionTrainer,
    RegressionTrainer,
)


class MLMetricMixin:
    """Mixin supplying the four common ML-metric helpers.

    Intended MRO: ``class Foo(MLMetricMixin, MetricBase): ...``
    so that the mixin methods are resolved first and subclasses can
    still override selectively (e.g. classification/regression each
    keep their own ``_validate_target``).
    """

    # ------------------------------------------------------------------
    # 1. Target validation (general-purpose version)
    # ------------------------------------------------------------------
    def _validate_target(self) -> tuple[list[str] | None, str | None]:
        """Validate target column(s).

        Returns:
            (target_cols, None) on success, or (None, error_message) on failure.
            Always returns a *list* of target column names, even for a single target.
        """
        dataset_spec = self._context.get("dataset_spec")
        if not dataset_spec or not hasattr(dataset_spec, "target_list"):
            return None, "No dataset spec available"

        targets: list[str] = dataset_spec.target_list
        if not targets:
            return None, "Target column not specified"

        missing_real = [t for t in targets if t not in self._real_data.columns]
        if missing_real:
            return None, f"Target columns not found in real data: {missing_real}"

        missing_synth = [t for t in targets if t not in self._synth_data.columns]
        if missing_synth:
            return None, f"Target columns not found in synthetic data: {missing_synth}"

        return targets, None

    # ------------------------------------------------------------------
    # 2. Feature-column selection
    # ------------------------------------------------------------------
    def _get_feature_cols(self, target_cols: str | list[str]) -> list[str]:
        """Get all feature columns (excluding target(s) and ID columns).

        CatBoost handles categorical columns natively, so we include all
        columns except the target(s) and any columns marked as ``id`` in the
        dataset spec schema.
        """
        if isinstance(target_cols, str):
            exclude: set[str] = {target_cols}
        else:
            exclude = set(target_cols)

        id_columns: set[str] = set()
        dataset_spec = self._context.get("dataset_spec")
        if dataset_spec and hasattr(dataset_spec, "schema"):
            for col_name, col_spec in dataset_spec.schema.items():
                if hasattr(col_spec, "column_type") and col_spec.column_type == "id":
                    id_columns.add(col_name)

        return [
            col for col in self._real_data.columns if col not in exclude and col not in id_columns
        ]

    # ------------------------------------------------------------------
    # 3. Task-type detection
    # ------------------------------------------------------------------
    def _determine_task_type(self, target_col: str) -> Literal["classification", "regression"]:
        """Determine task type based on target column characteristics.

        Priority:
            1. Explicit ``task_type`` in ``dataset_spec``
            2. Heuristic based on target column dtype / unique-value count
        """
        dataset_spec = self._context.get("dataset_spec")

        # Check explicit task_type
        if dataset_spec and hasattr(dataset_spec, "task_type") and dataset_spec.task_type:
            return dataset_spec.task_type

        # Heuristic inference
        target = self._real_data[target_col]

        # Non-numeric -> classification
        if not pd.api.types.is_numeric_dtype(target):
            return "classification"

        n_unique = target.nunique()
        n_samples = len(target)

        # Few unique values -> classification
        if n_unique <= 10:
            return "classification"

        # Integer with few unique values relative to samples -> classification
        if target.dtype.kind in "iu" and n_unique <= min(50, n_samples // 10):
            return "classification"

        return "regression"

    # ------------------------------------------------------------------
    # 4. Trainer factory
    # ------------------------------------------------------------------
    def _get_trainer(self, task_type: str, n_targets: int = 1):
        """Return the appropriate CatBoost trainer for *task_type*.

        When *n_targets* > 1 the multi-output variant is returned:
        - Regression  → CatBoost with ``loss_function='MultiRMSE'``
        - Classification → ``MultiOutputClassifier(CatBoostClassifier)``
        """
        if n_targets > 1:
            if task_type == "classification":
                return MultiTargetClassificationTrainer(n_trials=self.n_trials, n_runs=self.n_runs)
            return MultiTargetRegressionTrainer(n_trials=self.n_trials, n_runs=self.n_runs)
        if task_type == "classification":
            return ClassificationTrainer(n_trials=self.n_trials, n_runs=self.n_runs)
        return RegressionTrainer(n_trials=self.n_trials, n_runs=self.n_runs)
