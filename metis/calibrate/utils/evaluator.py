"""In-memory evaluation runner for calibration.

Uses the Orchestrator's public evaluate_dataframes() API to run evaluations
directly on DataFrames without any disk I/O.
"""

import math
from pathlib import Path

import pandas as pd
import yaml

from metis.shared.schema_utils import filter_schema_columns


class InMemoryEvaluator:
    """
    Runs METIS evaluation on in-memory DataFrames.

    Avoids all disk I/O by using Orchestrator.evaluate_dataframes() directly.
    Config template is loaded once and reused across iterations.
    """

    def __init__(self, config_template_path: str):
        """
        Initialize evaluator with config template.

        Args:
            config_template_path: Path to YAML config template
        """
        config_path = Path(config_template_path)
        with config_path.open(encoding="utf-8") as f:
            self.config_template = yaml.safe_load(f)

        # Disable reports for calibration runs
        self.config_template["report"] = {"formats": [], "output_dir": ""}

        # Disable calibration within calibration runs to avoid recursion
        self.config_template.pop("calibration", None)

        # Filter schema to only include evaluable columns
        schema = self.config_template.get("data", {}).get("schema", {})
        if schema:
            self.config_template["data"]["schema"] = filter_schema_columns(schema)

    def evaluate(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        seed: int,
        *,
        holdout: pd.DataFrame | None = None,
        families: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """
        Run evaluation and extract family scores AND per-metric values.

        ``holdout`` reaches the metrics as in a normal evaluation.
        ``families`` restricts the configured metrics to those families (the
        calibration never evaluates utility, and each lower reference only
        the family it anchors).

        Returns:
            Tuple of (family_scores, metric_values_by_family, raw_values_by_family).
            - family_scores: ``{"fidelity": 0.85, ...}``
            - metric_values_by_family: ``{"fidelity": {"metric_id": normalized, ...}, ...}``
              Normalized per-metric values used for aggregator tuning.
            - raw_values_by_family: ``{"fidelity": {"metric_id": raw, ...}, ...}``
              Raw (pre-normalization) per-metric values for traceability.
        """
        from metis.application.orchestrator import Orchestrator

        # Filter config schema to columns present in data
        config = self.config_template.copy()
        schema = config.get("data", {}).get("schema", {})
        if schema:
            data_columns = set(real_data.columns)
            config["data"] = {**config["data"]}
            config["data"]["schema"] = {
                col: spec for col, spec in schema.items() if col in data_columns
            }

        if families is not None:
            config["metrics"] = self.metrics_of(families)
        orchestrator = Orchestrator()
        run_summary = orchestrator.evaluate_dataframes(
            real_data=real_data,
            synth_data=synth_data,
            config=config,
            seed=seed,
            skip_reports=True,
            holdout=holdout,
        )

        # Extract family scores (aggregated by the Orchestrator's Aggregator)
        family_scores: dict[str, float] = {}
        aggregates = run_summary.aggregates
        for family in ["fidelity", "privacy", "utility"]:
            score_key = f"{family}_score"
            if score_key in aggregates:
                family_scores[family] = aggregates[score_key]

        # Extract per-metric normalized values grouped by family
        metric_values_by_family: dict[str, dict[str, float]] = {}
        # Extract per-metric raw (pre-normalization) values grouped by family
        raw_values_by_family: dict[str, dict[str, float]] = {}

        for result in run_summary.results:
            # NaN without ``error`` exists too (e.g. MIA without holdout):
            # it would turn every candidate aggregator's score into NaN.
            if getattr(result, "status", "ok") == "ok" and math.isfinite(float(result.value)):
                metric_values_by_family.setdefault(result.family, {})[result.id] = result.value
                raw_values_by_family.setdefault(result.family, {})[result.id] = (
                    self._extract_raw_value(result)
                )

        return family_scores, metric_values_by_family, raw_values_by_family

    def evaluate_results(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        seed: int,
        *,
        holdout: pd.DataFrame | None = None,
        metrics: list[str],
        ml_efficiency_config: dict | None = None,
    ) -> list:
        """The ``MetricResult`` list of *metrics*, details included.

        ``ml_efficiency_config`` entries override the template's ones (the
        utility signal probe sets ``scale_only`` and ``n_splits``).
        """
        from metis.application.orchestrator import Orchestrator

        config = self.config_template.copy()
        schema = config.get("data", {}).get("schema", {})
        if schema:
            config["data"] = {
                **config["data"],
                "schema": {c: s for c, s in schema.items() if c in real_data.columns},
            }
        config["metrics"] = list(metrics)
        if ml_efficiency_config:
            evaluation = dict(config.get("evaluation") or {})
            evaluation["ml_efficiency_config"] = {
                **(evaluation.get("ml_efficiency_config") or {}),
                **ml_efficiency_config,
            }
            config["evaluation"] = evaluation
        run_summary = Orchestrator().evaluate_dataframes(
            real_data=real_data,
            synth_data=synth_data,
            config=config,
            seed=seed,
            skip_reports=True,
            holdout=holdout,
        )
        return list(run_summary.results)

    def metrics_of(self, families: tuple[str, ...]) -> list[str]:
        """The configured metric ids that belong to *families* (shorthands expanded)."""
        from metis.domain.taxonomy import expand_metric_ids, get_metric_hierarchy

        configured = self.config_template.get("metrics") or self.config_template.get(
            "evaluation", {}
        ).get("metric_ids", [])
        return [
            m
            for m in expand_metric_ids(list(configured))
            if get_metric_hierarchy(m)["family"] in families
        ]

    @staticmethod
    def _extract_raw_value(result) -> float:
        """Extract the raw (pre-normalization) value from a MetricResult.

        Handles three patterns found in metric implementations:
        1. Per-column metrics (ks, wasserstein, etc.): mean of per-column raw_value.
        2. Global with explicit raw key (mmd → raw_mmd, energy_distance → raw_energy_distance).
        3. Global composite / pair-wise: no raw available, return the normalized value.
        """
        details = result.details

        # Pattern 1: per-column metrics with {col: {raw_value: ...}}
        per_col_raws = [
            v["raw_value"] for v in details.values() if isinstance(v, dict) and "raw_value" in v
        ]
        if per_col_raws:
            return sum(per_col_raws) / len(per_col_raws)

        # Pattern 2: global metric with raw_<metric_id> key
        raw_key = f"raw_{result.id}"
        if raw_key in details:
            return float(details[raw_key])

        # Pattern 3: no raw available — use the normalized value itself
        return float(result.value)
