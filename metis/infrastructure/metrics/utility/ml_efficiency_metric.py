"""
Unified ML Efficiency Metric.

Auto-detects classification vs regression and evaluates with configurable
strategies.  TTR (Train Real, Test Real) is computed **once** as the
baseline and reused by every strategy comparison.

Score formula per strategy:
    classification → delta = |F1_strategy − F1_ttr|
    regression     → delta = |MAE_strategy − MAE_ttr| / MAE_ttr

    utility = 1 − delta   (clamped to [0, 1])

For the aggregate ``ml_efficiency`` metric the deltas of every selected
strategy are averaged: ``1 − mean(deltas)``.
"""

from typing import Any

import numpy as np
import pandas as pd

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.base import MetricBase
from metis.infrastructure.metrics.registry import register
from metis.infrastructure.metrics.utility.ml_common import MLMetricMixin
from metis.infrastructure.metrics.utility.ml_efficiency.base import StrategyResult, TrainingStrategy
from metis.infrastructure.metrics.utility.ml_efficiency.trts import TRTSMetric
from metis.infrastructure.metrics.utility.ml_efficiency.tstr import TSTRMetric
from metis.infrastructure.metrics.utility.ml_efficiency.ttr import TTRMetric
from metis.infrastructure.metrics.utility.ml_efficiency.ttrs import TTRSMetric
from metis.infrastructure.metrics.utility.ml_efficiency.tts import TTSMetric

# Strategies the user can request (TTR is always the baseline, not selectable)
AVAILABLE_STRATEGIES: set[str] = {"tstr", "trts", "tts", "ttrs"}

_STRATEGY_METRICS: dict[TrainingStrategy, type] = {
    TrainingStrategy.TTR: TTRMetric,
    TrainingStrategy.TTS: TTSMetric,
    TrainingStrategy.TRTS: TRTSMetric,
    TrainingStrategy.TSTR: TSTRMetric,
    TrainingStrategy.TTRS: TTRSMetric,
}


# ── helpers ──────────────────────────────────────────────────────────────

# Key used to store the shared TTR result in the evaluation context dict.
# All standalone utility metrics (tts, tstr, trts, ttrs) read from this
# cache so that TTR is computed exactly **once** per evaluation run.
_TTR_CACHE_KEY = "_ttr_baseline_result"


def _compute_delta(
    strategy_score: float,
    ttr_score: float,
    task_type: str,
) -> float:
    """Return the normalised distance from the TTR baseline."""
    if task_type == "regression":
        return abs(strategy_score - ttr_score) / ttr_score if ttr_score > 0 else 1.0
    # classification – F1 already in [0, 1]
    return abs(strategy_score - ttr_score)


def _evaluate_strategy(
    strategy: TrainingStrategy,
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
    trainer: Any,
    seed: int,
) -> StrategyResult:
    """Prepare data for *strategy*, train, and return a ``StrategyResult``."""
    try:
        metric = _STRATEGY_METRICS[strategy]()
        data = metric.prepare_data(real, synth, target_col, feature_cols, seed=seed)

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
            raw_metric=mean_score,
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


def _get_or_compute_ttr(
    context: dict[str, Any],
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
    trainer: Any,
    seed: int,
) -> StrategyResult:
    """Return the cached TTR baseline, computing it on first call.

    The result is stored in ``context[_TTR_CACHE_KEY]`` so that every
    standalone utility metric in the same evaluation run shares a single,
    consistent TTR baseline.  This eliminates redundant Optuna
    optimisations (4×100 trials → 1×100 trials) and guarantees that all
    metrics compare against the exact same reference score.
    """
    cached: StrategyResult | None = context.get(_TTR_CACHE_KEY)
    if cached is not None:
        return cached

    ttr = _evaluate_strategy(
        TrainingStrategy.TTR,
        real,
        synth,
        target_col,
        feature_cols,
        trainer,
        seed,
    )
    context[_TTR_CACHE_KEY] = ttr
    return ttr


def _prepare_classification_labels(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, int | list[int]]:
    """Ensure target labels are usable by CatBoost; return n_classes.

    Handles both single-target (str) and multi-target (list[str]).
    CatBoost rejects ``pandas.arrays.StringArray``.  We cast the target
    column(s) to ``object`` dtype so that ``.values`` returns a plain
    numpy array that CatBoost accepts.
    """
    real = real.copy()
    synth = synth.copy()

    cols = [target_col] if isinstance(target_col, str) else list(target_col)
    n_classes_list: list[int] = []

    for col in cols:
        real[col] = real[col].astype(object)
        synth[col] = synth[col].astype(object)

        y_real = real[col].values
        if y_real.dtype.kind == "f":
            real[col] = y_real.astype(int)
            synth[col] = synth[col].values.astype(int)

        n_classes_list.append(int(len(np.unique(real[col].values))))

    n_classes = n_classes_list[0] if isinstance(target_col, str) else n_classes_list
    return real, synth, n_classes


# ── standalone strategy metrics ──────────────────────────────────────────


class BaseStandaloneStrategyMetric(MLMetricMixin, MetricBase):
    """Base for ``utility.tts``, ``utility.tstr``, etc.

    Computes TTR once, then computes the requested strategy and returns
    ``1 − delta``.
    """

    def __init__(self, n_trials: int = 100, n_runs: int = 1):
        super().__init__()
        self.n_trials = n_trials
        self.n_runs = n_runs

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> "BaseStandaloneStrategyMetric":
        self._setup(real_data, synth_data, context)
        return self

    def _compute_strategy(self, strategy: TrainingStrategy) -> MetricResult:
        metric_id = f"utility.{strategy.value}"
        try:
            target_cols, error = self._validate_target()
            if error:
                return self._error_result(metric_id, error)

            # Use the first target for task-type detection (all share task_type)
            task_type = self._determine_task_type(target_cols[0])
            feature_cols = self._get_feature_cols(target_cols)
            if not feature_cols:
                return self._error_result(metric_id, "No feature columns found")

            seed = self._context.get("seed", 42)
            # Single target → pass str for backward compat; multi → list
            target_arg: str | list[str] = target_cols[0] if len(target_cols) == 1 else target_cols
            trainer = self._get_trainer(task_type, n_targets=len(target_cols))

            real, synth = self._real_data, self._synth_data
            n_classes = None
            if task_type == "classification":
                real, synth, n_classes = _prepare_classification_labels(real, synth, target_arg)

            # 1. TTR baseline (shared across all standalone utility metrics)
            ttr = _get_or_compute_ttr(
                self._context,
                real,
                synth,
                target_arg,
                feature_cols,
                trainer,
                seed,
            )
            if not ttr.is_valid:
                return self._error_result(metric_id, f"TTR baseline failed: {ttr.error}")

            # 2. Strategy
            result = _evaluate_strategy(
                strategy, real, synth, target_arg, feature_cols, trainer, seed
            )
            if not result.is_valid:
                return self._error_result(metric_id, f"{strategy.value} failed: {result.error}")

            # 3. Score
            delta = _compute_delta(result.score, ttr.score, task_type)
            utility_score = max(0.0, min(1.0, 1.0 - delta))

            details: dict[str, Any] = {
                "task_type": task_type,
                "metric": "F1 (weighted)" if task_type == "classification" else "MAE",
                "model": "CatBoost",
                "strategy": strategy.value,
                "strategy_score": result.score,
                "ttr_baseline": ttr.score,
                "delta_from_ttr": delta,
                "std": result.std,
                "n_features": len(feature_cols),
                "n_targets": len(target_cols),
                "target_columns": target_cols,
                "best_params": result.best_params,
            }
            if n_classes is not None:
                details["n_classes"] = n_classes

            return MetricResult(
                id=metric_id,
                value=float(utility_score),
                details=details,
                family="utility",
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._error_result(metric_id, f"{strategy.value} computation failed: {e}")

    @staticmethod
    def _error_result(metric_id: str, error: str) -> MetricResult:
        return MetricResult(
            id=metric_id,
            value=float("nan"),
            details={"error": error},
            family="utility",
            purpose_tags=set(),
        )


# ── registered standalone metrics ────────────────────────────────────────


@register("utility.tts")
class TTSStandaloneMetric(BaseStandaloneStrategyMetric, TTSMetric):
    """Train on Synthetic, Test on Synthetic."""

    name: str = "tts"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "tts"}

    def compute(self) -> MetricResult:
        return self._compute_strategy(TrainingStrategy.TTS)


@register("utility.tstr")
class TSTRStandaloneMetric(BaseStandaloneStrategyMetric, TSTRMetric):
    """Train on Synthetic, Test on Real."""

    name: str = "tstr"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "tstr"}

    def compute(self) -> MetricResult:
        return self._compute_strategy(TrainingStrategy.TSTR)


@register("utility.trts")
class TRTSStandaloneMetric(BaseStandaloneStrategyMetric, TRTSMetric):
    """Train on Real, Test on Synthetic."""

    name: str = "trts"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "trts"}

    def compute(self) -> MetricResult:
        return self._compute_strategy(TrainingStrategy.TRTS)


@register("utility.ttrs")
class TTRSStandaloneMetric(BaseStandaloneStrategyMetric, TTRSMetric):
    """Train on Real+Synthetic, Test on Real."""

    name: str = "ttrs"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "ttrs"}

    def compute(self) -> MetricResult:
        return self._compute_strategy(TrainingStrategy.TTRS)


# ── aggregate metric ─────────────────────────────────────────────────────


@register("utility.ml_efficiency")
class MLEfficiencyMetric(MLMetricMixin, MetricBase):
    """Unified ML Efficiency Metric with auto task detection.

    TTR is computed **once** as the baseline.  Each selected strategy is
    evaluated and the final score is ``1 − mean(deltas)``.
    """

    name: str = "ml_efficiency"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "auto"}

    def __init__(
        self,
        strategies: list[str] | None = None,
        n_trials: int = 100,
        n_runs: int = 1,
    ):
        super().__init__()
        self.n_trials = n_trials
        self.n_runs = n_runs

        if strategies is None:
            self._selected_strategies = sorted(AVAILABLE_STRATEGIES)
        else:
            invalid = {s.lower() for s in strategies} - AVAILABLE_STRATEGIES
            if invalid:
                raise ValueError(
                    f"Invalid strategies: {invalid}. Available: {AVAILABLE_STRATEGIES}"
                )
            self._selected_strategies = [s.lower() for s in strategies]

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> "MLEfficiencyMetric":
        self._setup(real_data, synth_data, context)
        return self

    def compute(self) -> MetricResult:
        try:
            target_cols, error = self._validate_target()
            if error:
                return self._err(error)

            task_type = self._determine_task_type(target_cols[0])
            feature_cols = self._get_feature_cols(target_cols)
            if not feature_cols:
                return self._err("No feature columns found")

            seed = self._context.get("seed", 42)
            target_arg: str | list[str] = target_cols[0] if len(target_cols) == 1 else target_cols
            trainer = self._get_trainer(task_type, n_targets=len(target_cols))

            real, synth = self._real_data, self._synth_data
            n_classes = None
            if task_type == "classification":
                real, synth, n_classes = _prepare_classification_labels(real, synth, target_arg)

            # 1. TTR baseline (shared via context cache)
            ttr = _get_or_compute_ttr(
                self._context,
                real,
                synth,
                target_arg,
                feature_cols,
                trainer,
                seed,
            )
            if not ttr.is_valid:
                return self._err(f"TTR baseline failed: {ttr.error}")

            # 2. Evaluate selected strategies
            strategy_results: dict[str, StrategyResult] = {}
            for name in self._selected_strategies:
                strategy_results[name] = _evaluate_strategy(
                    TrainingStrategy(name),
                    real,
                    synth,
                    target_arg,
                    feature_cols,
                    trainer,
                    seed,
                )

            # 3. Compute deltas and final score
            deltas: dict[str, float] = {}
            valid_deltas: list[float] = []
            for name, result in strategy_results.items():
                if result.is_valid:
                    d = _compute_delta(result.score, ttr.score, task_type)
                    deltas[name] = d
                    valid_deltas.append(d)

            if not valid_deltas:
                return self._err("No valid strategy results")

            final_score = max(0.0, min(1.0, 1.0 - float(np.mean(valid_deltas))))

            # 4. Build details
            details: dict[str, Any] = {
                "task_type": task_type,
                "metric": "F1 (weighted)" if task_type == "classification" else "MAE",
                "model": "CatBoost",
                "n_features": len(feature_cols),
                "n_targets": len(target_cols),
                "target_columns": target_cols,
                "ttr_baseline": ttr.score,
                "ttr_std": ttr.std,
                "selected_strategies": self._selected_strategies,
                "delta_scores": deltas,
                "strategies": {},
            }
            if n_classes is not None:
                details["n_classes"] = n_classes

            for name, result in strategy_results.items():
                details["strategies"][name] = {
                    "score": result.score,
                    "std": result.std,
                    "delta_from_ttr": deltas.get(name, 0.0),
                    "is_valid": result.is_valid,
                    "error": result.error,
                }

            return MetricResult(
                id="utility.ml_efficiency",
                value=float(final_score),
                details=details,
                family="utility",
                purpose_tags=self.purpose_tags,
            )

        except Exception as e:
            return self._err(f"ML efficiency computation failed: {e}")

    def _err(self, msg: str) -> MetricResult:
        return MetricResult(
            id="utility.ml_efficiency",
            value=float("nan"),
            details={"error": msg},
            family="utility",
            purpose_tags=self.purpose_tags,
        )
