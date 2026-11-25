"""Step 5 — Execute all requested metrics.

Single responsibility: instantiate metrics from registry, feed them the
correct data slice (cat / num / full), and collect results.

Context handed to every metric's ``fit``:

* ``stats_store``, ``seed``, ``schema``, ``excluded_ids``, ``dataset_spec``,
  ``config`` (the full config dict);
* ``target`` — first target column (``data.target``) or ``None``;
* ``holdout`` — the real holdout table, preprocessed exactly like the real
  one (same fitted caster, same columns), as the ``full`` view
  (``pd.DataFrame``), or ``None`` when no holdout was given.  Only the
  ``full`` view is published, also for metrics routed to ``cat``/``num``.

Metric constructor arguments come from ``evaluation.metric_params``::

    evaluation:
      metric_params:
        privacy.k_anonymity: {quasi_identifiers: [age, zip], target_k: 3}

Each entry is checked against the constructor signature; an unknown key
makes that metric fail with a clear ``details.error`` (NaN value) instead of
being silently ignored.
"""

import inspect
from typing import Any

import pandas as pd

from ...domain.contracts import MetricRegistry
from ...domain.entities import DatasetSpec, EvalPlan, MetricResult, TransformedData
from ...domain.errors import SchemaError
from ...infrastructure.metrics.registry import get_metric_registry
from ...infrastructure.runtime.cache import StatsStore
from ...infrastructure.runtime.logging import get_logger


class MissingTargetError(SchemaError):
    """A utility metric was requested without ``data.target``."""


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
        *,
        config: dict[str, Any] | None = None,
        holdout: TransformedData | None = None,
        utility_signal: dict[str, Any] | None = None,
        privacy_tables: tuple[TransformedData, TransformedData] | None = None,
    ) -> list[MetricResult]:
        """Compute every metric in *plan* and return result list.

        Failed metrics produce a ``MetricResult`` with ``error`` in details
        instead of raising.
        ``utility_signal`` is the calibration's decision on whether the real
        table has signal for the utility metrics (``CalibrationBounds.utility_signal``).
        """
        stats_store = StatsStore()
        targets = spec.target_list if spec is not None else []
        context = {
            "stats_store": stats_store,
            "seed": seed,
            "schema": real.get_column_types(),
            "excluded_ids": real.excluded_ids,
            "dataset_spec": spec,
            # Metrics read their own run-time options from here
            # (e.g. ``evaluation.ml_efficiency_config``).
            "config": config or {},
            # First target column (``data.target``); None without target.
            "target": targets[0] if targets else None,
            # Real holdout preprocessed like the real table (``full`` view).
            "holdout": holdout.full if holdout is not None else None,
            # Signal decision of the utility metrics, from the calibration.
            "utility_signal": utility_signal,
        }

        results: list[MetricResult] = []
        for i, metric_id in enumerate(plan.metric_ids):
            # Privacy asks whether training rows were copied: it needs every
            # training and synthetic row, not the equal-size subsamples.
            if privacy_tables is not None and metric_id.startswith("privacy."):
                result = self._compute_single(metric_id, *privacy_tables, context)
            else:
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
            if metric_id.startswith("utility.") and not context.get("target"):
                raise MissingTargetError(
                    f"{metric_id} needs a target column: set 'data.target' in the config "
                    "(utility metrics train a model to predict it)."
                )
            metric_cls = self._registry.get(metric_id)
            real_df, synth_df = self._route_data(metric_cls, real, synth)
            params = self._metric_params(metric_id, metric_cls, context.get("config") or {})
            metric = metric_cls(**params)
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

    # ----- constructor parameters --------------------------------------------

    @staticmethod
    def _metric_params(metric_id: str, metric_cls: type, config: dict[str, Any]) -> dict[str, Any]:
        """Constructor kwargs for *metric_id* from ``evaluation.metric_params``.

        Raises ``ValueError`` (caught as the metric's ``details.error``) when
        a key is not a parameter of the metric's constructor.
        """
        all_params = (config.get("evaluation") or {}).get("metric_params") or {}
        params = all_params.get(metric_id) or {}
        if not params:
            return {}
        if not isinstance(params, dict):
            raise ValueError(
                f"evaluation.metric_params.{metric_id} must be a mapping, got {params!r}"
            )
        sig = inspect.signature(metric_cls.__init__)
        accepted = [
            name
            for name, p in sig.parameters.items()
            if name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
        unknown = sorted(set(params) - set(accepted))
        if unknown and not takes_kwargs:
            raise ValueError(
                f"Unknown parameter(s) {unknown} in evaluation.metric_params.{metric_id}. "
                f"Accepted: {accepted or 'none'}"
            )
        return dict(params)

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
