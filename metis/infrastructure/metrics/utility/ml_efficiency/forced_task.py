"""
Shared base of ``utility.classification_efficiency`` and
``utility.regression_efficiency``: ``utility.ml_efficiency`` with the task
forced to one type.

Kept apart from both metric modules because it must not import the metric
registry: the registry imports those modules, and importing either one
first used to fail with a circular import.
"""

from typing import Any

import pandas as pd

from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.base import MetricBase
from metis.infrastructure.metrics.utility.ml_common import DEFAULT_N_SPLITS, MLMetricMixin


class _ForcedTaskEfficiencyMetric(MLMetricMixin, MetricBase):
    """``utility.ml_efficiency`` restricted to one task type."""

    metric_id: str
    task_type: str

    def __init__(
        self,
        n_runs: int = 3,
        n_splits: int = DEFAULT_N_SPLITS,
        thread_count: int | None = None,
    ):
        """
        Args:
            n_runs: Number of refits per split
            n_splits: Number of train/test splits each strategy is scored on
            thread_count: CatBoost threads (``None`` = from ``METIS_N_JOBS``)
        """
        super().__init__()
        self.n_runs = n_runs
        self.n_splits = n_splits
        self.thread_count = thread_count
        self._strategies: list[str] | None = None

    def fit(
        self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: dict[str, Any]
    ) -> _ForcedTaskEfficiencyMetric:
        """Fit metric to data."""
        # Imported here: ``ml_efficiency_metric`` imports this package.
        from metis.infrastructure.metrics.utility.ml_efficiency_metric import (
            _parse_strategies,  # noqa: PLC0415
        )

        self._setup(real_data, synth_data, context)
        ml_config = self._apply_ml_config(context)
        self._strategies = _parse_strategies(ml_config.get("strategies"))
        context["_ttr_process_cache"] = bool(ml_config.get("cache_ttr", True))
        return self

    def compute(self) -> MetricResult:
        from metis.infrastructure.metrics.utility.ml_efficiency_metric import (  # noqa: PLC0415
            _error_result,
            _parse_strategies,
            run_ml_efficiency,
        )

        try:
            strategies = self._strategies or _parse_strategies(None)
            return run_ml_efficiency(
                self, strategies, self.metric_id, self.purpose_tags, task_type=self.task_type
            )
        except Exception as e:
            return _error_result(
                self.metric_id, f"{self.name} computation failed: {e}", self.purpose_tags
            )
