"""
Base classes for metric aggregation.

Provides common interfaces and functionality for aggregating metrics
across columns and categories using Stochastic Dominance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd

from .results import ColumnMetricResult


@dataclass
class AggregationResult:
    """
    Base result class for metric aggregation.

    Provides common structure and methods for aggregation results
    at any level (tails, scales, coverage, marginal, etc.).

    Attributes:
        score: Final aggregated score Q ∈ [0, 1] where 1 = best
        column_scores: μᵢ for each column
        metric_details: Full details per metric {metric_name: {column: ColumnMetricResult}}
        metrics_used: list of metric names used in aggregation
        n_columns: Number of columns processed
        n_metrics: Number of metrics computed
    """

    score: float
    column_scores: dict[str, float]
    metric_details: dict[str, dict[str, ColumnMetricResult]]
    metrics_used: list[str]
    n_columns: int
    n_metrics: int

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "score": self.score,
            "column_scores": self.column_scores,
            "metric_details": {
                metric: {col: res.to_dict() for col, res in cols.items()}
                for metric, cols in self.metric_details.items()
            },
            "metrics_used": self.metrics_used,
            "n_columns": self.n_columns,
            "n_metrics": self.n_metrics,
        }

    def get_report_data(self, metric_id: str, category: str) -> dict[str, Any]:
        """
        Get structured data for report generation.

        Args:
            metric_id: Identifier for the metric (e.g., "fidelity.marginal.tails")
            category: Category name for interpretation (e.g., "tails", "scales")

        Returns:
            Dictionary with all reporting data
        """
        return {
            "metric_id": metric_id,
            "score": self.score,
            "interpretation": self._interpret_score(category),
            "column_scores": self.column_scores,
            "worst_columns": self._get_worst_columns(5),
            "metrics_used": self.metrics_used,
            "summary_stats": {
                "n_columns": self.n_columns,
                "n_metrics": self.n_metrics,
                "mean_column_score": (
                    float(np.mean(list(self.column_scores.values()))) if self.column_scores else 0.0
                ),
            },
        }

    def _interpret_score(self, category: str) -> str:
        """
        Provide human-readable interpretation of the score.

        Args:
            category: Category name for the interpretation message

        Returns:
            Human-readable interpretation string
        """
        if self.score >= 0.9:
            return f"Excelente - Las distribuciones de {category} son muy similares"
        if self.score >= 0.7:
            return f"Bueno - Las distribuciones de {category} son razonablemente similares"
        if self.score >= 0.5:
            return f"Moderado - Algunas diferencias en las distribuciones de {category}"
        if self.score >= 0.3:
            return f"Pobre - Diferencias significativas en las distribuciones de {category}"
        return f"Muy pobre - Las distribuciones de {category} son muy diferentes"

    def _get_worst_columns(self, n: int = 5) -> list[dict[str, Any]]:
        """
        Get the n columns with worst scores.

        Args:
            n: Number of worst columns to return

        Returns:
            list of dictionaries with column name and score
        """
        sorted_cols = sorted(self.column_scores.items(), key=lambda x: x[1])
        return [{"column": col, "score": score} for col, score in sorted_cols[:n]]

    def get_per_metric_summary(self) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for each metric.

        Returns:
            Dictionary mapping metric names to their summary statistics
        """
        summary = {}
        for metric_name, col_results in self.metric_details.items():
            valid_values = [r.normalized_value for r in col_results.values() if r.is_valid]
            if valid_values:
                summary[metric_name] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "n_valid": len(valid_values),
                }
        return summary


# type variable for result types
R = TypeVar("R", bound=AggregationResult)


class BaseColumnAggregator(ABC, Generic[R]):
    """
    Abstract base class for column-level metric aggregators.

    Provides common functionality for aggregating multiple metrics
    across columns using Stochastic Dominance (FSD+SSD).

    type Parameters:
        R: The result type this aggregator produces (must extend AggregationResult)

    Subclasses must implement:
        - _get_metrics_registry(): Returns the registry of available metrics
        - _get_default_metrics(): Returns the list of default metric names
        - _create_result(): Creates the specific result type
    """

    def __init__(self, metrics: list[str] | None = None):
        """
        Initialize the aggregator.

        Args:
            metrics: list of metric names to use. If None, uses defaults.
        """
        registry = self._get_metrics_registry()
        defaults = self._get_default_metrics()

        if metrics is None:
            metrics = defaults.copy()

        for m in metrics:
            if m not in registry:
                available = list(registry.keys())
                raise ValueError(f"Unknown metric '{m}'. Available: {available}")

        self.metrics_to_use = metrics
        self._real_data: pd.DataFrame | None = None
        self._synth_data: pd.DataFrame | None = None
        self._metric_instances: dict[str, Any] = {}
        self._computed = False
        self._result: R | None = None

    @classmethod
    @abstractmethod
    def _get_metrics_registry(cls) -> dict[str, type]:
        """
        Get the registry of available metrics.

        Returns:
            Dictionary mapping metric names to their classes
        """
        pass

    @classmethod
    @abstractmethod
    def _get_default_metrics(cls) -> list[str]:
        """
        Get the default list of metrics to use.

        Returns:
            list of metric names
        """
        pass

    @abstractmethod
    def _create_result(
        self,
        score: float,
        column_scores: dict[str, float],
        metric_details: dict[str, dict[str, ColumnMetricResult]],
        n_columns: int,
        n_metrics: int,
    ) -> R:
        """
        Create the specific result type for this aggregator.

        Args:
            score: Final aggregated score
            column_scores: Scores per column
            metric_details: Detailed results per metric
            n_columns: Number of columns
            n_metrics: Number of metrics

        Returns:
            Result instance of the appropriate type
        """
        pass

    @classmethod
    def available_metrics(cls) -> list[str]:
        """Return list of available metric names."""
        return list(cls._get_metrics_registry().keys())

    def fit(self, real_data: pd.DataFrame, synth_data: pd.DataFrame) -> "BaseColumnAggregator[R]":
        """
        Initialize with data.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset

        Returns:
            Self for method chaining
        """
        self._real_data = real_data
        self._synth_data = synth_data
        self._computed = False

        # Initialize metric instances
        registry = self._get_metrics_registry()
        self._metric_instances = {}
        for metric_name in self.metrics_to_use:
            metric_class = registry[metric_name]
            metric = metric_class()
            metric.fit(real_data, synth_data)
            self._metric_instances[metric_name] = metric

        return self

    def compute(self) -> R:
        """
        Compute all metrics and aggregate into final score.

        Uses Stochastic Dominance (FSD+SSD) for aggregation.

        Returns:
            Result with score, column scores, and details
        """
        from metis.infrastructure.metrics.aggregation.stochastic_dominance import aggregate_metrics

        if self._real_data is None:
            raise ValueError("Must call fit() before compute()")

        # Step 1: Compute all metrics for all columns
        metric_details: dict[str, dict[str, ColumnMetricResult]] = {}

        for metric_name, metric in self._metric_instances.items():
            metric_details[metric_name] = metric.compute_all_columns()

        # Step 2: Collect all columns that have valid results
        all_columns = set()
        for metric_results in metric_details.values():
            for col, result in metric_results.items():
                if result.is_valid:
                    all_columns.add(col)

        all_columns = sorted(all_columns)
        n_cols = len(all_columns)
        n_metrics = len(self.metrics_to_use)

        if n_cols == 0:
            self._result = self._create_result(
                score=0.0,
                column_scores={},
                metric_details=metric_details,
                n_columns=0,
                n_metrics=n_metrics,
            )
            self._computed = True
            return self._result

        # Step 3: Create matrix A[column, metric]
        A = np.zeros((n_cols, n_metrics))
        for i, col in enumerate(all_columns):
            for j, metric_name in enumerate(self.metrics_to_use):
                if metric_name in metric_details:
                    metric_results = metric_details[metric_name]
                    if col in metric_results and metric_results[col].is_valid:
                        A[i, j] = metric_results[col].normalized_value

        # Step 4: Aggregate using Stochastic Dominance
        column_scores_array, final_score = aggregate_metrics(A)
        column_scores = {col: float(column_scores_array[i]) for i, col in enumerate(all_columns)}

        self._result = self._create_result(
            score=float(final_score),
            column_scores=column_scores,
            metric_details=metric_details,
            n_columns=n_cols,
            n_metrics=n_metrics,
        )

        self._computed = True
        return self._result

    @property
    def result(self) -> R | None:
        """Get the last computed result."""
        return self._result

    def get_column_breakdown(self) -> dict[str, dict[str, float]]:
        """
        Get detailed breakdown of scores per column per metric.

        Returns:
            Nested dict: {column: {metric: normalized_score}}
        """
        if not self._computed or self._result is None:
            raise ValueError("Must call compute() first")

        breakdown = {}
        for col in self._result.column_scores:
            breakdown[col] = {}
            for metric_name, metric_results in self._result.metric_details.items():
                if col in metric_results and metric_results[col].is_valid:
                    breakdown[col][metric_name] = metric_results[col].normalized_value

        return breakdown

    def get_worst_columns(self, n: int = 5) -> list[tuple]:
        """
        Get the n columns with worst scores.

        Args:
            n: Number of worst columns to return

        Returns:
            list of (column_name, score) tuples, sorted by score ascending
        """
        if not self._computed or self._result is None:
            raise ValueError("Must call compute() first")

        sorted_cols = sorted(self._result.column_scores.items(), key=lambda x: x[1])
        return sorted_cols[:n]

    def get_worst_metrics_per_column(self, column: str) -> list[tuple]:
        """
        Get metrics sorted by score for a specific column.

        Args:
            column: Column name

        Returns:
            list of (metric_name, normalized_score) tuples, sorted ascending
        """
        if not self._computed or self._result is None:
            raise ValueError("Must call compute() first")

        scores = []
        for metric_name, metric_results in self._result.metric_details.items():
            if column in metric_results and metric_results[column].is_valid:
                scores.append((metric_name, metric_results[column].normalized_value))

        return sorted(scores, key=lambda x: x[1])
