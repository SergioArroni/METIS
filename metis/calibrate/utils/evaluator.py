"""In-memory evaluation runner for calibration.

Uses the Orchestrator's public evaluate_dataframes() API to run evaluations
directly on DataFrames without any disk I/O.
"""

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
        self, real_data: pd.DataFrame, synth_data: pd.DataFrame, seed: int
    ) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """
        Run evaluation and extract family scores AND per-metric values.

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

        orchestrator = Orchestrator()
        run_summary = orchestrator.evaluate_dataframes(
            real_data=real_data,
            synth_data=synth_data,
            config=config,
            seed=seed,
            skip_reports=True,
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
            if "error" not in result.details:
                metric_values_by_family.setdefault(result.family, {})[result.id] = result.value
                raw_values_by_family.setdefault(result.family, {})[result.id] = (
                    self._extract_raw_value(result)
                )

        return family_scores, metric_values_by_family, raw_values_by_family

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
