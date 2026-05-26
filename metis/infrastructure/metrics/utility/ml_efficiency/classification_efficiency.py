"""
Classification Efficiency Metric.

Evaluates synthetic data utility for classification tasks using
all 5 training strategies and comparing against TTR baseline.
"""

from typing import Any

import numpy as np
import pandas as pd

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.base import MetricBase
from metis.infrastructure.metrics.registry import register
from metis.infrastructure.metrics.utility.ml_common import MLMetricMixin

from .base import MLEfficiencyResult, StrategyResult, TrainingStrategy
from .catboost_trainer import ClassificationTrainer
from .trts import TRTSMetric
from .tstr import TSTRMetric
from .ttr import TTRMetric
from .ttrs import TTRSMetric
from .tts import TTSMetric


@register("utility.classification_efficiency")
class ClassificationEfficiencyMetric(MLMetricMixin, MetricBase):
    """
    Classification ML Efficiency Metric using F1 Score.

    Evaluates synthetic data utility through 5 training strategies:
    - TTR: Train Real, Test Real (baseline)
    - TTS: Train Synthetic, Test Synthetic
    - TRTS: Train Real, Test Synthetic
    - TSTR: Train Synthetic, Test Real
    - TTRS: Train Real+Synthetic, Test Real

    Final score = 1 - mean(|strategy_F1 - TTR_F1|) for strategies TTS, TRTS, TSTR, TTRS.

    Uses CatBoost with Optuna hyperparameter optimization based on
    Gorishniy et al., 2021 configuration.
    """

    name: str = "classification_efficiency"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "classification", "f1"}

    def __init__(self, n_trials: int = 100, n_runs: int = 3):
        """
        Initialize metric.

        Args:
            n_trials: Number of Optuna trials for hyperparameter optimization
            n_runs: Number of evaluation runs with best parameters
        """
        super().__init__()
        self.n_trials = n_trials
        self.n_runs = n_runs

        # Initialize strategy metrics
        self._strategies = {
            TrainingStrategy.TTR: TTRMetric(),
            TrainingStrategy.TTS: TTSMetric(),
            TrainingStrategy.TRTS: TRTSMetric(),
            TrainingStrategy.TSTR: TSTRMetric(),
            TrainingStrategy.TTRS: TTRSMetric(),
        }

    def fit(
        self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: dict[str, Any]
    ) -> "ClassificationEfficiencyMetric":
        """Fit metric to data."""
        self._setup(real_data, synth_data, context)

        # Allow override from config for calibration optimization
        ml_config = context.get("config", {}).get("evaluation", {}).get("ml_efficiency_config", {})
        if ml_config:
            self.n_trials = ml_config.get("n_trials", self.n_trials)
            self.n_runs = ml_config.get("n_runs", self.n_runs)

        return self

    def _validate_target(self) -> tuple[str | None, str | None]:
        """Validate target column for classification. Returns (target_col, error)."""
        dataset_spec = self._context.get("dataset_spec")
        target_col = None

        if dataset_spec and hasattr(dataset_spec, "target_list"):
            targets = dataset_spec.target_list
            target_col = targets[0] if targets else None

        if not target_col or target_col not in self._real_data.columns:
            return None, "Target column not specified or not found"

        y_real = self._real_data[target_col].values
        if hasattr(y_real, "dtype") and pd.api.types.is_string_dtype(y_real):
            y_real = np.array(y_real, dtype=str)
        n_classes = len(np.unique(y_real))

        if n_classes > min(50, len(y_real) // 10):
            return (
                None,
                f"Target has too many unique values ({n_classes}) for classification",
            )

        if n_classes < 2:
            return None, f"Target has only {n_classes} unique values, need at least 2"

        if target_col not in self._synth_data.columns:
            return None, "Target column not found in synthetic data"

        return target_col, None

    def _evaluate_strategy(
        self,
        strategy: TrainingStrategy,
        target_col: str,
        feature_cols: list[str],
        seed: int,
    ) -> StrategyResult:
        """Evaluate a single training strategy."""
        strategy_metric = self._strategies[strategy]
        trainer = ClassificationTrainer(n_trials=self.n_trials, n_runs=self.n_runs)

        try:
            # Prepare data for this strategy
            data = strategy_metric.prepare_data(
                self._real_data,
                self._synth_data,
                target_col,
                feature_cols,
                seed=seed,
            )

            # Train and evaluate
            best_params, scores, mean_score, std_score = trainer.train_and_evaluate(
                data["X_train"],
                data["X_test"],
                data["y_train"],
                data["y_test"],
                seed=seed,
            )

            return StrategyResult(
                strategy=strategy,
                score=mean_score,
                raw_metric=mean_score,  # F1 is the raw metric
                scores_per_run=scores,
                std=std_score,
                best_params=best_params,
                is_valid=True,
            )

        except Exception as e:
            return StrategyResult(
                strategy=strategy,
                score=0.0,
                raw_metric=0.0,
                scores_per_run=[],
                std=0.0,
                best_params={},
                is_valid=False,
                error=str(e),
            )

    def _compute_final_score(
        self,
        strategy_results: dict[TrainingStrategy, StrategyResult],
    ) -> tuple[float, dict[str, float]]:
        """
        Compute final score based on comparison with TTR baseline.

        Score = 1 - mean(|strategy_F1 - TTR_F1|) for TTS, TRTS, TSTR, TTRS.
        """
        ttr_result = strategy_results[TrainingStrategy.TTR]

        if not ttr_result.is_valid:
            return 0.0, {}

        baseline_f1 = ttr_result.score
        delta_scores = {}

        # Compute deltas for each non-baseline strategy
        comparison_strategies = [
            TrainingStrategy.TTS,
            TrainingStrategy.TRTS,
            TrainingStrategy.TSTR,
            TrainingStrategy.TTRS,
        ]

        valid_deltas = []
        for strategy in comparison_strategies:
            result = strategy_results.get(strategy)
            if result and result.is_valid:
                delta = abs(result.score - baseline_f1)
                delta_scores[strategy.value] = delta
                valid_deltas.append(delta)

        if not valid_deltas:
            return 0.0, delta_scores

        # Final score: 1 - mean(deltas), clamped to [0, 1]
        mean_delta = np.mean(valid_deltas)
        final_score = max(0.0, min(1.0, 1.0 - mean_delta))

        return final_score, delta_scores

    def compute(self) -> MetricResult:
        """Compute classification efficiency metric."""
        try:
            # Validate target
            target_col, error = self._validate_target()
            if error:
                return MetricResult(
                    id="utility.classification_efficiency",
                    value=float("nan"),
                    details={"error": error},
                    family="utility",
                    purpose_tags=self.purpose_tags,
                )

            # Get features (including categorical - CatBoost handles them natively)
            feature_cols = self._get_feature_cols(target_col)
            if not feature_cols:
                return MetricResult(
                    id="utility.classification_efficiency",
                    value=float("nan"),
                    details={"error": "No feature columns found"},
                    family="utility",
                    purpose_tags=self.purpose_tags,
                )

            seed = self._context.get("seed", 42)
            n_classes = int(len(np.unique(self._real_data[target_col].values)))

            # Convert StringArray labels to proper numpy arrays
            y_real = self._real_data[target_col].values
            y_synth = self._synth_data[target_col].values
            if hasattr(y_real, "dtype") and pd.api.types.is_string_dtype(y_real):
                self._real_data = self._real_data.copy()
                self._synth_data = self._synth_data.copy()
                self._real_data[target_col] = np.array(y_real, dtype=str)
                self._synth_data[target_col] = np.array(y_synth, dtype=str)
            elif y_real.dtype.kind == "f":
                self._real_data = self._real_data.copy()
                self._synth_data = self._synth_data.copy()
                self._real_data[target_col] = y_real.astype(int)
                self._synth_data[target_col] = y_synth.astype(int)

            # Evaluate all strategies
            strategy_results: dict[TrainingStrategy, StrategyResult] = {}
            for strategy in TrainingStrategy:
                strategy_results[strategy] = self._evaluate_strategy(
                    strategy, target_col, feature_cols, seed
                )

            # Compute final score
            final_score, delta_scores = self._compute_final_score(strategy_results)

            # Build result
            result = MLEfficiencyResult(
                final_score=final_score,
                baseline_score=strategy_results[TrainingStrategy.TTR].score,
                strategy_results=strategy_results,
                delta_scores=delta_scores,
                task_type="classification",
                metric_name="f1",
                n_features=len(feature_cols),
                n_classes=n_classes,
            )

            # Build details for MetricResult
            details = {
                "task_type": "classification",
                "metric": "F1 (weighted)",
                "model": "CatBoost",
                "optimization": "Optuna",
                "n_trials": self.n_trials,
                "n_runs": self.n_runs,
                "n_classes": n_classes,
                "n_features": len(feature_cols),
                "baseline_ttr_f1": result.baseline_score,
                "delta_scores": delta_scores,
                "strategies": {
                    s.value: {
                        "f1": r.score,
                        "f1_std": r.std,
                        "delta_from_ttr": delta_scores.get(s.value, 0.0),
                        "is_valid": r.is_valid,
                        "error": r.error,
                    }
                    for s, r in strategy_results.items()
                },
            }

            return MetricResult(
                id="utility.classification_efficiency",
                value=float(final_score),
                details=details,
                family="utility",
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return MetricResult(
                id="utility.classification_efficiency",
                value=float("nan"),
                details={"error": f"Classification efficiency computation failed: {str(e)}"},
                family="utility",
                purpose_tags=self.purpose_tags,
            )
