"""Step 6 — Aggregate metric results into family / composite scores.

Single responsibility: take a flat list of MetricResult objects and produce
an aggregates dict ready for RunSummary.

Delegates the heavy lifting to the existing ``Aggregator`` class in
``metis.application.aggregator`` so that hierarchical scoring, calibration
normalisation, and tuned aggregation functions are reused without duplication.
"""

from typing import Any

from ...domain.contracts import BoundsStorage
from ...domain.entities import EvalPlan, MetricResult
from ...infrastructure.runtime.logging import get_logger
from ..aggregator import Aggregator


class ResultAggregator:
    """Thin façade that configures and invokes ``Aggregator``."""

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    def aggregate(
        self,
        results: list[MetricResult],
        plan: EvalPlan,
        config: dict[str, Any],
        bounds: BoundsStorage | None = None,
    ) -> dict[str, Any]:
        """Build aggregated scores.

        Args:
            results: Output from MetricEvaluator.
            plan: The evaluation plan used.
            config: Full config dict (reads ``aggregation`` / ``calibration``).
            bounds: Pre-loaded calibration bounds (overrides config path).

        Returns:
            Aggregates dict suitable for ``RunSummary.aggregates``.
        """
        agg_cfg = config.get("aggregation", {})
        cal_cfg = config.get("calibration", {})

        risk_aversion = agg_cfg.get("risk_aversion", 5.0)
        weights = agg_cfg.get("weights")

        # Resolve calibration bounds: prefer injected, then config path
        bounds_path = cal_cfg.get("bounds_file") if bounds is None else None
        aggregators_path = cal_cfg.get("aggregators_file")

        aggregator = Aggregator(
            risk_aversion=risk_aversion,
            calibration_bounds_path=bounds_path,
            optimal_aggregators_path=aggregators_path,
            calibration_bounds=bounds,
        )

        aggregates = aggregator.aggregate(results, plan, weights)
        self._logger.info(
            "Aggregated %d metrics → composite=%.4f",
            len(results),
            aggregates.get("composite_score", 0.0),
        )
        return aggregates
