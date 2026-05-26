"""
Regression Efficiency Metric.

Evaluates synthetic data utility for regression tasks using
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
from .catboost_trainer import RegressionTrainer
from .trts import TRTSMetric
from .tstr import TSTRMetric
from .ttr import TTRMetric
from .ttrs import TTRSMetric
from .tts import TTSMetric


@register("utility.regression_efficiency")
class RegressionEfficiencyMetric(MLMetricMixin, MetricBase):
    """
    Regression ML Efficiency Metric using MAE.

    Evaluates synthetic data utility through 5 training strategies:
    - TTR: Train Real, Test Real (baseline)
    - TTS: Train Synthetic, Test Synthetic
    - TRTS: Train Real, Test Synthetic
    - TSTR: Train Synthetic, Test Real
    - TTRS: Train Real+Synthetic, Test Real

    For MAE (lower is better), we compute:
    delta = |strategy_MAE - TTR_MAE| / TTR_MAE (normalized relative difference)
    Final score = 1 - mean(deltas), clamped to [0, 1]

    Uses CatBoost with Optuna hyperparameter optimization based on
    Gorishniy et al., 2021 configuration.
    """

    name: str = "regression_efficiency"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "regression", "mae"}

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
    ) -> "RegressionEfficiencyMetric":
        """Fit metric to data."""
        self._setup(real_data, synth_data, context)
        return self

    def _validate_target(self) -> tuple[str | None, str | None]:
        """Validate target column for regression. Returns (target_col, error)."""
        dataset_spec = self._context.get("dataset_spec")
        target_col = None

        if dataset_spec and hasattr(dataset_spec, "target_list"):
            targets = dataset_spec.target_list
            target_col = targets[0] if targets else None

        if not target_col or target_col not in self._real_data.columns:
            return None, "Target column not specified or not found"

        y_real = self._real_data[target_col]

        if not pd.api.types.is_numeric_dtype(y_real):
            return None, "Target column is not numeric"

        n_unique = y_real.nunique()
        if n_unique <= 10:
            return (
                None,
                f"Target has only {n_unique} unique values. Use classification.",
            )

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
        trainer = RegressionTrainer(n_trials=self.n_trials, n_runs=self.n_runs)

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
                score=mean_score,  # For regression, this is MAE
                raw_metric=mean_score,
                scores_per_run=scores,
                std=std_score,
                best_params=best_params,
                is_valid=True,
            )

        except Exception as e:
            return StrategyResult(
                strategy=strategy,
                score=float("inf"),  # Invalid MAE
                raw_metric=float("inf"),
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

        For MAE: delta = |strategy_MAE - TTR_MAE| / TTR_MAE (normalized)
        Score = 1 - mean(deltas), clamped to [0, 1]
        """
        ttr_result = strategy_results[TrainingStrategy.TTR]

        if not ttr_result.is_valid or ttr_result.score <= 0:
            return 0.0, {}

        baseline_mae = ttr_result.score
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
                # Normalized absolute difference
                delta = abs(result.score - baseline_mae) / baseline_mae
                delta_scores[strategy.value] = delta
                valid_deltas.append(delta)

        if not valid_deltas:
            return 0.0, delta_scores

        # Final score: 1 - mean(deltas), clamped to [0, 1]
        mean_delta = np.mean(valid_deltas)
        final_score = max(0.0, min(1.0, 1.0 - mean_delta))

        return final_score, delta_scores

    def compute(self) -> MetricResult:
        """Compute regression efficiency metric."""
        try:
            # Validate target
            target_col, error = self._validate_target()
            if error:
                return MetricResult(
                    id="utility.regression_efficiency",
                    value=float("nan"),
                    details={"error": error},
                    family="utility",
                    purpose_tags=self.purpose_tags,
                )

            # Get features (including categorical - CatBoost handles them natively)
            feature_cols = self._get_feature_cols(target_col)
            if not feature_cols:
                return MetricResult(
                    id="utility.regression_efficiency",
                    value=float("nan"),
                    details={"error": "No feature columns found"},
                    family="utility",
                    purpose_tags=self.purpose_tags,
                )

            seed = self._context.get("seed", 42)

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
                task_type="regression",
                metric_name="mae",
                n_features=len(feature_cols),
            )

            # Build details for MetricResult
            details = {
                "task_type": "regression",
                "metric": "MAE",
                "model": "CatBoost",
                "optimization": "Optuna",
                "n_trials": self.n_trials,
                "n_runs": self.n_runs,
                "n_features": len(feature_cols),
                "baseline_ttr_mae": result.baseline_score,
                "delta_scores": delta_scores,
                "strategies": {
                    s.value: {
                        "mae": r.score,
                        "mae_std": r.std,
                        "delta_from_ttr": delta_scores.get(s.value, 0.0),
                        "is_valid": r.is_valid,
                        "error": r.error,
                    }
                    for s, r in strategy_results.items()
                },
            }

            return MetricResult(
                id="utility.regression_efficiency",
                value=float(final_score),
                details=details,
                family="utility",
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return MetricResult(
                id="utility.regression_efficiency",
                value=float("nan"),
                details={"error": f"Regression efficiency computation failed: {str(e)}"},
                family="utility",
                purpose_tags=self.purpose_tags,
            )
