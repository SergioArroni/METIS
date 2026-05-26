"""
Base classes and types for ML Efficiency metrics.

Provides data types and abstract interfaces for training strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class TrainingStrategy(Enum):
    """Training/Testing strategy for ML efficiency evaluation."""

    TTR = "ttr"  # Train on Real, Test on Real (baseline)
    TTS = "tts"  # Train on Synthetic, Test on Synthetic
    TRTS = "trts"  # Train on Real, Test on Synthetic
    TSTR = "tstr"  # Train on Synthetic, Test on Real
    TTRS = "ttrs"  # Train on Real+Synthetic, Test on Real


@dataclass
class StrategyResult:
    """Result from a single training strategy evaluation."""

    strategy: TrainingStrategy
    score: float  # Primary metric (F1 for classification, 1-MAE_normalized for regression)
    raw_metric: float  # Raw metric value (F1 or MAE)
    scores_per_run: list[float]  # Scores from multiple runs
    std: float  # Standard deviation across runs
    best_params: dict[str, Any]  # Best hyperparameters from optimization
    is_valid: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy.value,
            "score": self.score,
            "raw_metric": self.raw_metric,
            "scores_per_run": self.scores_per_run,
            "std": self.std,
            "best_params": self.best_params,
            "is_valid": self.is_valid,
            "error": self.error,
        }


@dataclass
class MLEfficiencyResult:
    """Aggregated result from ML efficiency evaluation."""

    final_score: float  # Aggregated score in [0, 1]
    baseline_score: float  # TTR score (reference)
    strategy_results: dict[TrainingStrategy, StrategyResult]
    delta_scores: dict[str, float]  # |strategy - TTR| for each strategy
    task_type: str  # "classification" or "regression"
    metric_name: str  # "f1" or "mae"
    n_features: int
    n_classes: int | None = None  # Only for classification
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_score": self.final_score,
            "baseline_score": self.baseline_score,
            "strategy_results": {k.value: v.to_dict() for k, v in self.strategy_results.items()},
            "delta_scores": self.delta_scores,
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            **self.details,
        }


def subsample(
    X: pd.DataFrame, y: np.ndarray, n_samples: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Subsample a dataset to n_samples if it exceeds that size."""
    if len(X) > n_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), size=n_samples, replace=False)
        return X.iloc[indices].reset_index(drop=True), y[indices]
    return X, y


def balance_datasets(
    X_real: pd.DataFrame,
    y_real: np.ndarray,
    X_synth: pd.DataFrame,
    y_synth: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Balance real and synthetic datasets to equal size for fair comparison."""
    n_samples = min(len(X_real), len(X_synth))
    X_real, y_real = subsample(X_real, y_real, n_samples, seed)
    X_synth, y_synth = subsample(X_synth, y_synth, n_samples, seed)
    return X_real, y_real, X_synth, y_synth


class BaseStrategyMetric(ABC):
    """
    Abstract base class for individual training strategy metrics.

    Each strategy (TTR, TTS, TRTS, TSTR, TTRS) implements this interface
    to provide consistent evaluation across different train/test combinations.
    """

    strategy: TrainingStrategy

    @abstractmethod
    def prepare_data(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        target_col: str | list[str],
        feature_cols: list[str],
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """
        Prepare train/test splits for this strategy.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            target_col: Target column name
            feature_cols: Feature column names
            test_size: Test set proportion
            seed: Random seed

        Returns:
            Dictionary with X_train, X_test, y_train, y_test arrays
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return human-readable description of this strategy."""
        pass
        pass
