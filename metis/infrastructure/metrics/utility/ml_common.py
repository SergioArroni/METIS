"""
Shared mixin for ML efficiency metrics.

Provides canonical implementations of helper methods that were previously
duplicated across BaseStandaloneStrategyMetric, MLEfficiencyMetric,
ClassificationEfficiencyMetric, and RegressionEfficiencyMetric.

Classes that mix this in are expected to also inherit from MetricBase
(which provides ``_context``, ``_real_data``, ``_synth_data``) and to
set ``self.n_runs`` before calling ``_get_trainer``.
"""

import warnings
from typing import Any, Literal

import pandas as pd

from metis.infrastructure.metrics.utility.ml_efficiency.catboost_trainer import (
    ClassificationTrainer,
    MultiTargetClassificationTrainer,
    MultiTargetRegressionTrainer,
    OrdinalClassificationTrainer,
    RegressionTrainer,
)

# Keys accepted in ``evaluation.ml_efficiency_config``.  ``scale_only`` is set
# by the calibration: it stops after measuring the real table's signal
# (``metis.calibrate.core.utility_signal``).
ML_CONFIG_KEYS: frozenset[str] = frozenset(
    {"n_runs", "n_splits", "thread_count", "strategies", "cache_ttr", "scale_only"}
)

# Accepted for old configs and ignored: there is no hyperparameter search any
# more (CatBoost defaults with early stopping, see ``catboost_trainer``).
DEPRECATED_ML_CONFIG_KEYS: frozenset[str] = frozenset({"n_trials"})

# ``_validate_target`` error when no target is configured: the metric does
# not apply (``MetricResult.status == "not_applicable"``), it did not fail.
NO_TARGET = "Target column not specified"

# Train/test splits each strategy is scored on (repeated 80/20 holdout).
DEFAULT_N_SPLITS = 5


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
        targets: list[str] = []
        if dataset_spec is not None and hasattr(dataset_spec, "target_list"):
            targets = list(dataset_spec.target_list)
        if not targets:
            # Fallback: the evaluation context may carry the target directly.
            context_target = self._context.get("target")
            if isinstance(context_target, str):
                targets = [context_target]
            elif context_target:
                targets = list(context_target)
        if not targets:
            return None, NO_TARGET
        # A datetime target is cast to ``<name>__ts``; an ordinal one keeps
        # ``<name>`` (its categorical form, the label to predict).
        targets = [
            f"{t}__ts"
            if t not in self._real_data.columns and f"{t}__ts" in self._real_data.columns
            else t
            for t in targets
        ]

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
        dataset spec schema.  An ordinal predictor enters with both of its
        views, the level (categorical) and the rank ``<name>__ord``: the model
        gets both, as everywhere in METIS (see ``SimpleCaster._cast_ordinal``).
        With both, CatBoost tells a broken order from a correct one better
        than with the level alone (paired t 4.1 vs 2.6 on cardio), and the
        rank alone misses a one-level shift.  The rank of an ordinal *target*
        is dropped: it is the label.
        """
        targets = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        exclude: set[str] = set(targets)
        exclude |= {f"{t}{suffix}" for t in targets for suffix in ("__ord", "__ts")}

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
    def _get_trainer(
        self,
        task_type: str,
        n_targets: int = 1,
        level_ranks: dict[int, float] | None = None,
    ):
        """Return the appropriate CatBoost trainer for *task_type*.

        When *n_targets* > 1 the multi-output variant is returned:
        - Regression  → CatBoost with ``loss_function='MultiRMSE'``
        - Classification → one ``CatBoostClassifier`` per target
        ``task_type == "ordinal"`` (a single ordinal target) is a classifier
        scored with QWK on the level ranks *level_ranks* (class code → rank).
        """
        kwargs = {
            "n_runs": self.n_runs,
            "thread_count": getattr(self, "thread_count", None),
        }
        if task_type == "ordinal":
            return OrdinalClassificationTrainer(level_ranks=level_ranks or {}, **kwargs)
        if n_targets > 1:
            if task_type == "classification":
                return MultiTargetClassificationTrainer(**kwargs)
            return MultiTargetRegressionTrainer(**kwargs)
        if task_type == "classification":
            return ClassificationTrainer(**kwargs)
        return RegressionTrainer(**kwargs)

    # ------------------------------------------------------------------
    # 5. Run-time configuration
    # ------------------------------------------------------------------
    def _apply_ml_config(self, context: dict[str, Any]) -> dict[str, Any]:
        """Override the training budget from ``evaluation.ml_efficiency_config``.

        Recognised keys: ``n_runs`` (refits per split, unrelated to
        ``evaluation.n_runs``), ``n_splits`` (train/test splits each strategy
        is scored on), ``thread_count``.  ``n_trials`` is accepted with a
        warning and ignored (there is no search any more).  Unknown keys
        raise, so a typo cannot silently fall back to a default.  Returns
        the section for subclass-specific keys.
        """
        ml_config = (context.get("config") or {}).get("evaluation", {}).get(
            "ml_efficiency_config"
        ) or {}
        unknown = set(ml_config) - ML_CONFIG_KEYS - DEPRECATED_ML_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Unknown keys in evaluation.ml_efficiency_config: {sorted(unknown)}. "
                f"Allowed: {sorted(ML_CONFIG_KEYS)}"
            )
        for key in sorted(DEPRECATED_ML_CONFIG_KEYS & set(ml_config)):
            warnings.warn(
                f"ml_efficiency_config.{key} is ignored: utility uses CatBoost defaults "
                "with early stopping, not a hyperparameter search",
                DeprecationWarning,
                stacklevel=2,
            )
        for key in ("n_runs", "n_splits", "thread_count"):
            if key in ml_config:
                value = ml_config[key]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"ml_efficiency_config.{key} must be an int, got {value!r}")
                if key != "thread_count" and value < 1:
                    raise ValueError(f"ml_efficiency_config.{key} must be >= 1, got {value!r}")
                if key == "thread_count" and value != -1 and value < 1:
                    # CatBoost rejects 0 and other negatives at fit time, which
                    # used to surface as a failed strategy (NaN), not as a config error.
                    raise ValueError(
                        f"ml_efficiency_config.thread_count must be -1 or >= 1, got {value!r}"
                    )
                setattr(self, key, value)
        return ml_config
