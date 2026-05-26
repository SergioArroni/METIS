"""Orchestrator — thin pipeline runner that chains contract-based steps.

Responsibilities (and nothing else):
  1. Load configuration.
  2. Determine single-run vs multi-run mode.
  3. For each seed: Load → Preprocess → Validate → Calibrate → Evaluate
     → Aggregate → Report.
  4. In multi-run mode: compute cross-run statistics and final report.

All heavy logic lives in the individual pipeline steps.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from ..domain.contracts import MetricRegistry
from ..domain.entities import EvalPlan, RunSummary
from ..domain.errors import ConfigError
from ..infrastructure.runtime.config import load_config
from ..infrastructure.runtime.logging import get_logger
from ..shared import get_seed_for_run, set_global_seed
from ..shared.multi_run_stats import calculate_multi_run_statistics
from .pipeline.aggregator import ResultAggregator
from .pipeline.calibrator import CalibrationStep
from .pipeline.evaluator import MetricEvaluator
from .pipeline.loader import DataLoader
from .pipeline.preprocessor import DataPreprocessor
from .pipeline.reporter import ReportGenerator
from .pipeline.validator import DataValidator


class Orchestrator:
    """Facade that composes pipeline steps via dependency injection.

    Each step satisfies a Protocol contract defined in ``metis.domain.contracts``.
    Steps may be replaced in tests by providing alternative implementations.
    """

    def __init__(
        self,
        metric_registry: MetricRegistry | None = None,
        loader: DataLoader | None = None,
        preprocessor: DataPreprocessor | None = None,
        validator: DataValidator | None = None,
        calibrator: CalibrationStep | None = None,
        evaluator: MetricEvaluator | None = None,
        aggregator: ResultAggregator | None = None,
        reporter: ReportGenerator | None = None,
    ):
        self.logger = get_logger(__name__)

        # Pipeline steps (defaults when not injected)
        self._loader = loader or DataLoader()
        self._preprocessor = preprocessor or DataPreprocessor()
        self._validator = validator or DataValidator()
        self._calibrator = calibrator or CalibrationStep()
        self._evaluator = evaluator or MetricEvaluator(metric_registry)
        self._aggregator = aggregator or ResultAggregator()
        self._reporter = reporter or ReportGenerator()

    # =====================================================================
    # Public API
    # =====================================================================

    def run(self, config_path: str) -> RunSummary:
        """Execute full pipeline from a YAML config file.

        Supports multi-run mode (``evaluation.n_runs > 1``) for
        statistical stability analysis across seeds.
        """
        self.logger.info("Starting evaluation with config: %s", config_path)
        config = load_config(config_path)

        repro = config.get("reproducibility", {})
        base_seed = repro.get("seed", 42)
        eval_cfg = config.get("evaluation", {})
        n_runs = eval_cfg.get("n_runs", 1)

        if n_runs < 1:
            raise ValueError(f"Invalid n_runs: {n_runs}. Must be >= 1.")

        if n_runs == 1:
            set_global_seed(base_seed)
            return self._run_single(config_path, config, base_seed)

        return self._run_multi(config_path, config, base_seed, n_runs)

    def evaluate_dataframes(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        config: dict[str, Any],
        seed: int = 42,
        skip_reports: bool = True,
        config_path: str | None = None,
    ) -> RunSummary:
        """Run evaluation on pre-loaded DataFrames (no file I/O).

        Public API used by calibration, benchmarks, and programmatic callers.

        Args:
            real_data: Real dataset.
            synth_data: Synthetic dataset.
            config: Full METIS configuration dict.
            seed: Random seed.
            skip_reports: Whether to skip report generation.
            config_path: Optional path to the YAML config file, required for
                auto-calibration via CacheManager. When provided and the config
                has a ``calibration`` section, calibration bounds are computed
                (or loaded from cache) and applied.
        """
        plan = self._build_plan(config)

        # Use loader for spec only (no disk I/O)
        _, _, spec = self._loader.load_from_dataframes(real_data, synth_data, config)

        # Preprocess (strips IDs, NaN removal, SimpleCaster)
        real_tf, synth_tf = self._preprocessor.preprocess(real_data, synth_data, config, spec)

        # Validate
        self._validator.validate(real_tf, synth_tf, spec, plan)

        # Calibrate if configured (uses transformed data for fingerprinting)
        bounds = self._calibrator.calibrate_if_needed(real_tf.full, config, config_path)

        # Evaluate
        results = self._evaluator.evaluate(plan, real_tf, synth_tf, spec, seed)

        # Aggregate
        aggregates = self._aggregator.aggregate(results, plan, config, bounds)

        summary = RunSummary(
            plan=plan,
            results=results,
            aggregates=aggregates,
            artifacts={
                "warnings": [],
                "schema_applied": real_tf.get_column_types(),
                "excluded_id_columns": real_tf.excluded_ids,
                "seed": seed,
            },
        )

        if not skip_reports:
            self._reporter.generate(summary, config)

        return summary

    # =====================================================================
    # Single-run pipeline
    # =====================================================================

    def _run_single(
        self,
        config_path: str,
        config: dict[str, Any],
        seed: int,
        skip_reports: bool = False,
    ) -> RunSummary:
        """Execute one pass of the evaluation pipeline."""

        # 1. Plan
        plan = self._build_plan(config)

        # 2. Load
        real_raw, synth_raw, spec = self._loader.load(config)

        # 3. Preprocess (strips IDs, NaN removal, SimpleCaster)
        real_tf, synth_tf = self._preprocessor.preprocess(real_raw, synth_raw, config, spec)

        # 4. Validate
        self._validator.validate(real_tf, synth_tf, spec, plan)

        # 5. Calibrate (uses transformed data for fingerprinting)
        bounds = self._calibrator.calibrate_if_needed(real_tf.full, config, config_path)

        # 6. Evaluate metrics
        results = self._evaluator.evaluate(plan, real_tf, synth_tf, spec, seed)

        # 7. Aggregate
        aggregates = self._aggregator.aggregate(results, plan, config, bounds)

        # 8. Build summary
        summary = RunSummary(
            plan=plan,
            results=results,
            aggregates=aggregates,
            artifacts={
                "config_path": config_path,
                "schema_applied": real_tf.get_column_types(),
                "excluded_id_columns": real_tf.excluded_ids,
                "seed": seed,
            },
        )

        # 9. Report
        if not skip_reports:
            self._reporter.generate(summary, config)

        self.logger.info("Evaluation completed — seed=%d", seed)
        return summary

    # =====================================================================
    # Multi-run pipeline
    # =====================================================================

    def _run_multi(
        self,
        config_path: str,
        config: dict[str, Any],
        base_seed: int,
        n_runs: int,
    ) -> RunSummary:
        """Execute multiple runs and aggregate cross-run statistics."""
        self.logger.info(
            "Multi-run: %d runs, seeds %d–%d", n_runs, base_seed, base_seed + n_runs - 1
        )

        report_cfg = config.get("report", {})
        runs_dir = Path(report_cfg.get("output_dir", "reports/evaluation_run")) / "__runs__"
        runs_dir.mkdir(parents=True, exist_ok=True)

        summaries: list[RunSummary] = []
        scores_per_run: list[dict[str, Any]] = []

        for idx in range(n_runs):
            run_seed = get_seed_for_run(base_seed, idx)
            self.logger.info("=== Run %d/%d (seed=%d) ===", idx + 1, n_runs, run_seed)
            set_global_seed(run_seed)

            summary = self._run_single(config_path, config, run_seed, skip_reports=True)
            summaries.append(summary)

            # Persist individual run
            self._reporter.save_run_json(summary, runs_dir / f"run_{idx}.json")
            scores_per_run.append(self._reporter.extract_scores(summary))

        # Cross-run statistics
        multi_stats = calculate_multi_run_statistics(scores_per_run)

        # Build aggregated summary from first run as template
        template = summaries[0]
        artifacts = {
            **template.artifacts,
            "multi_run_stats": multi_stats,
            "n_runs": n_runs,
            "base_seed": base_seed,
            "seeds_used": [get_seed_for_run(base_seed, i) for i in range(n_runs)],
        }

        final = RunSummary(
            plan=template.plan,
            results=template.results,
            aggregates=template.aggregates,
            artifacts=artifacts,
        )

        self._reporter.generate(final, config)
        self.logger.info("Multi-run evaluation completed")
        return final

    # =====================================================================
    # Plan construction
    # =====================================================================

    @staticmethod
    def _build_plan(config: dict[str, Any]) -> EvalPlan:
        """Extract an EvalPlan from the config dict.

        Raises:
            ConfigError: If ``metric_ids`` is missing or invalid.
        """
        eval_cfg = config.get("evaluation", {})
        # Priority: top-level metrics → evaluation.metric_ids (legacy)
        metric_ids = config.get("metrics") or eval_cfg.get("metric_ids")

        if not metric_ids:
            raise ConfigError("'metrics' must be specified in config")
        if not isinstance(metric_ids, list):
            raise ConfigError("'metrics' must be a list")

        # Expand category / subcategory shorthands into concrete metric IDs
        from metis.domain.taxonomy import expand_metric_ids

        metric_ids = expand_metric_ids(metric_ids)

        # Seed priority: evaluation.seed → reproducibility.seed → 42
        repro_seed = config.get("reproducibility", {}).get("seed", 42)
        plan_seed = eval_cfg.get("seed", repro_seed)

        return EvalPlan(
            metric_ids=metric_ids,
            seed=plan_seed,
            cv_splits=eval_cfg.get("cv_splits", 3),
        )
