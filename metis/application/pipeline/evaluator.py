"""Step 5 — Execute all requested metrics.

Single responsibility: instantiate metrics from registry, feed them the
correct data slice (cat / num / full), and collect results.
"""

from typing import Any

import pandas as pd

from ...domain.contracts import MetricRegistry
from ...domain.entities import DatasetSpec, EvalPlan, MetricResult, TransformedData
from ...infrastructure.metrics.registry import get_metric_registry
from ...infrastructure.runtime.cache import StatsStore
from ...infrastructure.runtime.logging import get_logger


class MetricEvaluator:
    """Runs every metric listed in the EvalPlan and returns results."""

    def __init__(self, metric_registry: MetricRegistry | None = None) -> None:
        self._registry = metric_registry or get_metric_registry()
        self._logger = get_logger(__name__)

    def evaluate(
        self,
        plan: EvalPlan,
        real: TransformedData,
        synth: TransformedData,
        spec: DatasetSpec,
        seed: int,
    ) -> list[MetricResult]:
        """Compute every metric in *plan* and return result list.

        Failed metrics produce a ``MetricResult`` with ``error`` in details
        instead of raising.
        """
        stats_store = StatsStore()
        context = {
            "stats_store": stats_store,
            "seed": seed,
            "schema": real.get_column_types(),
            "excluded_ids": real.excluded_ids,
            "dataset_spec": spec,
        }

        results: list[MetricResult] = []
        for i, metric_id in enumerate(plan.metric_ids):
            result = self._compute_single(metric_id, real, synth, context)
            results.append(result)
            if (i + 1) % 10 == 0:
                self._logger.debug("Completed %d/%d metrics", i + 1, len(plan.metric_ids))

        self._logger.info(
            "Completed %d metrics (%d ok, %d failed)",
            len(results),
            sum(1 for r in results if "error" not in r.details),
            sum(1 for r in results if "error" in r.details),
        )
        return results

    # ----- single metric execution -------------------------------------------

    def _compute_single(
        self,
        metric_id: str,
        real: TransformedData,
        synth: TransformedData,
        context: dict[str, Any],
    ) -> MetricResult:
        try:
            metric_cls = self._registry.get(metric_id)
            real_df, synth_df = self._route_data(metric_cls, real, synth)
            metric = metric_cls()
            metric.fit(real_df, synth_df, context)
            result = metric.compute()
            self._logger.debug("Metric %s = %.4f", metric_id, result.value)
            return result
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Metric %s failed: %s — will be excluded from aggregation", metric_id, exc
            )
            family = metric_id.split(".")[0] if "." in metric_id else "fidelity"
            return MetricResult(
                id=metric_id,
                value=float("nan"),
                details={"error": str(exc), "error_type": type(exc).__name__},
                family=family,
                purpose_tags=set(),
            )

    # ----- data routing ------------------------------------------------------

    @staticmethod
    def _route_data(
        metric_cls: type,
        real: TransformedData,
        synth: TransformedData,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Select the correct data slice based on metric requirements."""
        family = getattr(metric_cls, "family", None)
        if family in ("privacy", "utility"):
            return real.full, synth.full

        requires = getattr(metric_cls, "requires_data", "both")
        if requires == "cat":
            return real.cat, synth.cat
        if requires == "num":
            return real.num, synth.num
        return real.full, synth.full
