"""Orchestrator — thin pipeline runner that chains contract-based steps.

Responsibilities (and nothing else):
  1. Load configuration.
  2. Determine single-run vs multi-run mode.
  3. For each seed: Load → Preprocess → Validate → Calibrate → Evaluate
     → Aggregate → Report.
  4. In multi-run mode: compute cross-run statistics and final report.

All heavy logic lives in the individual pipeline steps.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metis.shared.provenance import metis_provenance

from ..domain.contracts import MetricRegistry
from ..domain.entities import DatasetSpec, EvalPlan, MetricResult, RunSummary, TransformedData
from ..domain.errors import ConfigError, SchemaError
from ..infrastructure.constraints import validate as validate_constraints
from ..infrastructure.metrics.registry import get_metric_registry
from ..infrastructure.runtime.config import load_config
from ..infrastructure.runtime.logging import get_logger
from ..shared import get_seed_for_run, set_global_seed
from ..shared.multi_run_stats import calculate_multi_run_statistics
from .pipeline.aggregator import ResultAggregator
from .pipeline.calibrator import CalibrationStep
from .pipeline.evaluator import MetricEvaluator
from .pipeline.loader import DataLoader
from .pipeline.preprocessor import DataPreprocessor, ReferenceContext
from .pipeline.reporter import ReportGenerator
from .pipeline.validator import DataValidator


def resolve_auto_schema(
    config: dict[str, Any], real: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``data.schema: auto`` replaced by the types the Phase −1 rules infer on
    *real* (``metis.infrastructure.typing.infer_schema``).

    Returns the config to use and, when the schema was inferred, the inferred
    schema with its notes (published in the artifacts).  Ordinal order cannot
    be inferred from the values: declare ordinals explicitly (``metis
    infer-schema`` writes a schema to review).
    """
    data = config.get("data") or {}
    if data.get("schema") != "auto":
        return config, None
    from metis.infrastructure.typing import infer_schema

    inferred = infer_schema(real)
    resolved = {**config, "data": {**data, "schema": inferred.schema}}
    return resolved, {"schema": inferred.schema, "notes": inferred.notes}


def equal_sizes(
    real: TransformedData, synth: TransformedData, seed: int
) -> tuple[TransformedData, TransformedData]:
    """The two tables at the size of the smaller one.

    The larger is subsampled at random (without replacement, rows kept in
    their order) with *seed*; every view (``cat``, ``num``, ``full``) keeps the
    same rows.  Equal sizes already pass through unchanged.
    """
    n = min(len(real.full), len(synth.full))

    def shrink(td: TransformedData, salt: int) -> TransformedData:
        if len(td.full) == n:
            return td
        rng = np.random.default_rng([int(seed), salt])
        idx = np.sort(rng.choice(len(td.full), n, replace=False))

        def pick(df: pd.DataFrame) -> pd.DataFrame:
            # A view without columns (e.g. no categorical column) has no rows.
            if len(df) != len(td.full):
                return df
            return df.iloc[idx].reset_index(drop=True)

        return TransformedData(
            cat=pick(td.cat),
            num=pick(td.num),
            full=pick(td.full),
            meta=dict(td.meta),
            excluded_ids=list(td.excluded_ids),
            schema=td.schema,
        )

    return shrink(real, 1), shrink(synth, 2)


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
        self._registry = metric_registry

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

        # Same seed priority as ``evaluate_dataframes`` and the SDK
        # (``evaluation.seed`` → ``reproducibility.seed`` → 42); this path read
        # only ``reproducibility.seed`` and ignored ``evaluation.seed``.
        base_seed = self._build_plan(config).seed
        eval_cfg = config.get("evaluation") or {}
        n_runs = eval_cfg.get("n_runs", 1)

        if n_runs < 1:
            raise ValueError(f"Invalid n_runs: {n_runs}. Must be >= 1.")

        if n_runs == 1:
            set_global_seed(base_seed)
            return self._run_single(config_path, config, base_seed)

        return self._run_multi(config_path, config, base_seed, n_runs)

    def prepare_reference(
        self, real_data: pd.DataFrame, config: dict[str, Any]
    ) -> ReferenceContext:
        """Preprocess the real table once, to reuse across ``evaluate_dataframes`` calls.

        The result depends only on *real_data* and ``config["data"]``; pass
        it as ``reference=`` with the same real table and data config.
        """
        spec = self._loader.build_spec(config)
        return self._preprocessor.prepare_reference(real_data, config, spec)

    def evaluate_dataframes(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        config: dict[str, Any],
        seed: int | None = None,
        skip_reports: bool = True,
        config_path: str | None = None,
        *,
        holdout: pd.DataFrame | None = None,
        reference: ReferenceContext | None = None,
        utility_signal: dict[str, Any] | None = None,
    ) -> RunSummary:
        """Run evaluation on pre-loaded DataFrames (no file I/O).

        Public API used by calibration, benchmarks, and programmatic callers.

        Args:
            real_data: Real dataset.
            synth_data: Synthetic dataset.
            config: Full METIS configuration dict.
            seed: Random seed.  ``None`` takes it from the config
                (``evaluation.seed`` → ``reproducibility.seed`` → 42), the
                same priority as the file-based pipeline.
            skip_reports: Whether to skip report generation.
            config_path: Optional path to the YAML config file, required for
                auto-calibration via CacheManager. When provided and the config
                has a ``calibration`` section, calibration bounds are computed
                (or loaded from cache) and applied.
            holdout: Optional real rows *not* used to train the generator.
                Preprocessed exactly like *real_data* (same fitted caster,
                same columns) and handed to the metrics as
                ``context["holdout"]`` (``full`` view).
            reference: Optional :class:`ReferenceContext` from
                :meth:`prepare_reference` for this same *real_data* and
                ``config["data"]``; skips re-preprocessing the real table.
            utility_signal: Optional decision on whether *real_data* has signal
                for the utility metrics, measured once for the table
                (``Evaluator.measure_utility_signal``); it takes precedence
                over the calibration's.  Without either, each run decides.
        """
        plan = self._build_plan(config)
        if seed is None:
            seed = plan.seed

        # Use loader for spec only (no disk I/O)
        _, _, spec = self._loader.load_from_dataframes(real_data, synth_data, config)

        summary, _ = self._evaluate(
            plan,
            config,
            spec,
            seed,
            real_raw=real_data,
            synth_raw=synth_data,
            holdout_raw=holdout,
            reference=reference,
            config_path=config_path,
            extra_artifacts={},
            utility_signal=utility_signal,
        )

        if not skip_reports:
            self._reporter.generate(summary, config)

        return summary

    # =====================================================================
    # Shared core: preprocess → validate → calibrate → evaluate → aggregate
    # =====================================================================

    def _evaluate(
        self,
        plan: EvalPlan,
        config: dict[str, Any],
        spec: DatasetSpec,
        seed: int,
        *,
        real_raw: pd.DataFrame,
        synth_raw: pd.DataFrame,
        holdout_raw: pd.DataFrame | None,
        reference: ReferenceContext | None,
        config_path: str | None,
        extra_artifacts: dict[str, Any],
        utility_signal: dict[str, Any] | None = None,
    ) -> tuple[RunSummary, Any]:
        self._check_metric_params(config, plan)
        config, inferred = resolve_auto_schema(config, real_raw)
        if inferred is not None:
            extra_artifacts = {**extra_artifacts, "inferred_schema": inferred}

        # Preprocess (strips IDs, NaN removal, SimpleCaster fitted on real)
        if reference is None:
            reference = self._preprocessor.prepare_reference(real_raw, config, spec)
        real_tf = reference.fresh_real()
        synth_tf = self._preprocessor.transform_other(synth_raw, reference, label="synth")
        holdout_tf = None
        if holdout_raw is not None:
            holdout_tf = self._preprocessor.transform_other(holdout_raw, reference, label="holdout")

        # Validate
        warnings = list(self._validator.validate(real_tf, synth_tf, spec, plan) or [])
        if holdout_tf is not None:
            self._check_holdout(real_tf, holdout_tf)

        # Domain rules (the transversal axis): checked row by row on the whole
        # synthetic table as given, reported per rule, never part of a score.
        constraints_report, constraint_warnings = validate_constraints(real_raw, synth_raw, config)
        warnings.extend(constraint_warnings)
        if constraints_report is not None:
            extra_artifacts = {**extra_artifacts, "constraints": constraints_report}

        # Real and synthetic are compared at the same size: the larger one is
        # subsampled at random to the size of the smaller (metric values depend
        # on the number of rows; author's rule, 2026-10-19).  Privacy keeps the
        # whole tables: subsampling the training rows would drop rows the
        # synthetic table copied and hide the leak (an exact copy of a fifth
        # of the training rows scored DCR 0.76 instead of ≈ 0).
        rows_in = {"real": len(real_tf.full), "synth": len(synth_tf.full)}
        privacy_tables = (real_tf, synth_tf)
        real_tf, synth_tf = equal_sizes(real_tf, synth_tf, seed)

        # Calibrate if configured.  The strategies run the whole pipeline on
        # real halves and noise, so they get the real table *before* casting
        # (see ``ReferenceContext.real_clean``).
        bounds = self._calibrator.calibrate_if_needed(reference.real_clean, config, config_path)
        if bounds is not None:
            warnings.extend(
                self._calibration_size_warning(bounds, len(real_tf.full), len(synth_tf.full))
            )

        # Evaluate
        results = self._evaluator.evaluate(
            plan,
            real_tf,
            synth_tf,
            spec,
            seed,
            config=config,
            holdout=holdout_tf,
            utility_signal=utility_signal or getattr(bounds, "utility_signal", None),
            privacy_tables=privacy_tables,
        )

        # Aggregate
        aggregates = self._aggregator.aggregate(results, plan, config, bounds)

        rows, row_warnings = self._row_artifacts(real_tf, synth_tf, holdout_tf)
        rows["rows_compared"] = {"real": len(real_tf.full), "synth": len(synth_tf.full)}
        rows["rows_before_equalising"] = rows_in
        warnings.extend(row_warnings)
        summary = RunSummary(
            plan=plan,
            results=results,
            aggregates=aggregates,
            artifacts={
                **extra_artifacts,
                "warnings": warnings,
                "schema_applied": real_tf.get_column_types(),
                "excluded_id_columns": real_tf.excluded_ids,
                "seed": seed,
                **metis_provenance(),
                **rows,
            },
        )
        return summary, bounds

    @staticmethod
    def _row_artifacts(
        real_tf: TransformedData,
        synth_tf: TransformedData,
        holdout_tf: TransformedData | None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Missing and invalid values of each table, published in the artifacts.

        No row is dropped any more.  ``synth_cells_invalid`` and
        ``synth_rows_with_invalid`` count synthetic values the caster could
        not read (a level that does not exist, an unparsable number...): they
        are treated as missing, the rules axis checks the values against the
        schema, and this count is the direct evidence.  ``synth_rows_dropped`` (and
        its fraction) stay at 0 for readers of the former accounting.
        """
        stats = {"real": real_tf.meta.get("_rows", {}), "synth": synth_tf.meta.get("_rows", {})}
        if holdout_tf is not None:
            stats["holdout"] = holdout_tf.meta.get("_rows", {})
        warnings = []
        for label, name in (("synth", "synthetic"), ("real", "real"), ("holdout", "holdout")):
            part = stats.get(label) or {}
            cells = part.get("cells_invalid", 0)
            if cells:
                warnings.append(
                    f"{cells} {name} value(s) in {part.get('rows_with_invalid', 0)} of "
                    f"{part.get('rows_in', 0)} rows could not be read and are treated as "
                    f"missing: {part.get('invalid_by_column', {})}"
                )
        synth = stats["synth"]
        artifacts = {
            "rows": stats,
            "synth_cells_invalid": int(synth.get("cells_invalid", 0)),
            "synth_rows_with_invalid": int(synth.get("rows_with_invalid", 0)),
            "synth_rows_dropped": 0,
            "synth_rows_dropped_fraction": 0.0,
        }
        if holdout_tf is not None:
            artifacts["holdout_rows"] = int(len(holdout_tf.full))
        return artifacts, warnings

    @staticmethod
    def _calibration_size_warning(bounds: Any, n_real: int, n_synth: int) -> list[str]:
        """Warn when the tables compared are larger than the calibration's.

        The anchors are measured on thirds of the real table, and metric
        values improve with the number of rows (cardio, real against real:
        fidelity 0.96 with 1000 rows, 0.99 with 9000; wasserstein 0.81 and
        0.97).  On larger tables the upper anchor is too low and calibrated
        scores come out optimistic.  The benchmark evaluates thirds too.
        """
        rows, size = bounds.get_metadata("dataset_rows"), bounds.get_metadata("sample_size")
        if not isinstance(rows, int) or not isinstance(size, int):
            return []
        third = min(size, rows // 5)  # holdout / unseen part of the 60/20/20 split
        if third <= 0 or min(n_real, n_synth) <= 1.25 * third:
            return []
        return [
            f"calibration anchors were measured on tables of {third} rows and this evaluation "
            f"compares {n_real} real with {n_synth} synthetic rows: metric values improve with "
            "the number of rows, so the calibrated scores are optimistic. Evaluate tables of "
            "the calibration size (as the benchmark does with its thirds) or read the "
            "uncalibrated scores."
        ]

    @staticmethod
    def _check_holdout(real_tf: TransformedData, holdout_tf: TransformedData) -> None:
        if holdout_tf.full.empty:
            raise SchemaError("Holdout data is empty after preprocessing")
        if list(holdout_tf.full.columns) != list(real_tf.full.columns):
            raise SchemaError(
                "Holdout columns differ from the real ones after preprocessing: "
                f"{sorted(set(holdout_tf.full.columns) ^ set(real_tf.full.columns))}"
            )

    def _check_metric_params(self, config: dict[str, Any], plan: EvalPlan) -> None:
        """Every metric id in the plan and in ``evaluation.metric_params`` is registered.

        An unknown id in ``metrics`` (a typo such as ``fidelity.tvdd``) used to
        run as a NaN metric that the aggregation then skipped: the evaluation
        "succeeded" without the metric that was asked for.
        """
        registry = self._registry or get_metric_registry()
        known = set(registry.list_ids())
        unknown_plan = [m for m in plan.metric_ids if m not in known]
        if unknown_plan:
            raise ConfigError(
                f"Unknown metric id(s) in 'metrics': {unknown_plan}. "
                "Use registered ids or family / category shorthands "
                "(e.g. 'fidelity', 'fidelity.marginal')."
            )
        metric_params = (config.get("evaluation") or {}).get("metric_params") or {}
        if not metric_params:
            return
        if not isinstance(metric_params, dict):
            raise ConfigError("'evaluation.metric_params' must map metric ids to mappings")
        unknown = sorted(set(metric_params) - known)
        if unknown:
            raise ConfigError(f"Unknown metric id(s) in evaluation.metric_params: {unknown}")
        unused = sorted(set(metric_params) - set(plan.metric_ids))
        if unused:
            self.logger.warning(
                "evaluation.metric_params has entries for metrics not in the plan: %s", unused
            )

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
        summary, _ = self._run_single_with_bounds(config_path, config, seed, skip_reports)
        return summary

    def _run_single_with_bounds(
        self,
        config_path: str,
        config: dict[str, Any],
        seed: int,
        skip_reports: bool = False,
    ) -> tuple[RunSummary, Any]:
        plan = self._build_plan(config)
        real_raw, synth_raw, spec = self._loader.load(config)
        holdout_raw = self._loader.load_holdout(config)

        summary, bounds = self._evaluate(
            plan,
            config,
            spec,
            seed,
            real_raw=real_raw,
            synth_raw=synth_raw,
            holdout_raw=holdout_raw,
            reference=None,
            config_path=config_path,
            extra_artifacts={"config_path": config_path},
        )

        if not skip_reports:
            self._reporter.generate(summary, config)

        self.logger.info("Evaluation completed — seed=%d", seed)
        return summary, bounds

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
        bounds = None

        for idx in range(n_runs):
            run_seed = get_seed_for_run(base_seed, idx)
            self.logger.info("=== Run %d/%d (seed=%d) ===", idx + 1, n_runs, run_seed)
            set_global_seed(run_seed)

            summary, bounds = self._run_single_with_bounds(
                config_path, config, run_seed, skip_reports=True
            )
            summaries.append(summary)

            # Persist individual run
            self._reporter.save_run_json(summary, runs_dir / f"run_{idx}.json")
            scores_per_run.append(self._reporter.extract_scores(summary))

        # Cross-run statistics
        multi_stats = calculate_multi_run_statistics(scores_per_run)

        # The summary describes all runs, not the first one: each metric is
        # the mean of its finite per-run values, and the aggregates are
        # recomputed from those means.
        plan = summaries[0].plan
        mean_results = self._mean_results(summaries)
        aggregates = self._aggregator.aggregate(mean_results, plan, config, bounds)
        artifacts = {
            **summaries[0].artifacts,
            "multi_run_stats": multi_stats,
            "n_runs": n_runs,
            "base_seed": base_seed,
            "seeds_used": [get_seed_for_run(base_seed, i) for i in range(n_runs)],
            "summary_mode": "mean_of_runs",
        }

        final = RunSummary(
            plan=plan,
            results=mean_results,
            aggregates=aggregates,
            artifacts=artifacts,
        )

        self._reporter.generate(final, config)
        self.logger.info("Multi-run evaluation completed")
        return final

    @staticmethod
    def _mean_results(summaries: list[RunSummary]) -> list[MetricResult]:
        """Per-metric mean over runs (finite values only), keeping run values."""
        by_id: dict[str, list[MetricResult]] = {}
        for summary in summaries:
            for result in summary.results:
                by_id.setdefault(result.id, []).append(result)

        merged: list[MetricResult] = []
        for metric_id, runs in by_id.items():
            values = [float(r.value) for r in runs]
            finite = [v for v in values if math.isfinite(v)]
            mean = float(np.mean(finite)) if finite else float("nan")
            template = next((r for r in runs if math.isfinite(float(r.value))), runs[0])
            details = dict(template.details)
            if not finite and "error" not in details:
                details["error"] = "metric was not finite in any run"
            details["multi_run"] = {
                "values": values,
                "mean": mean,
                "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "n_finite": len(finite),
            }
            merged.append(
                MetricResult(
                    id=metric_id,
                    value=mean,
                    details=details,
                    family=runs[0].family,
                    purpose_tags=set(runs[0].purpose_tags),
                )
            )
        return merged

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
