"""
Benchmark orchestrator for comparing multiple synthetic data generators.

This module coordinates the benchmarking process:
1. Load benchmark configuration
2. Instantiate and run multiple generators
3. Evaluate each with METIS using specified config
4. Aggregate results across generators and seeds
5. Generate comparison reports
"""

import copy
import gc
import json
import logging
import math
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from metis.application.orchestrator import Orchestrator as MetisOrchestrator
from metis.application.pipeline.preprocessor import DataPreprocessor
from metis.calibrate.cache.cache_manager import CacheManager
from metis.calibrate.cache.fingerprint import (
    compute_config_fingerprint,
    compute_data_fingerprint,
    generate_cache_key,
)
from metis.domain.entities import DatasetSpec
from metis.infrastructure.io.loaders import load_csv
from metis.shared.config_utils import none_safe as _none_safe
from metis.shared.schema_utils import extract_column_types
from metis.shared.splits import DEFAULT_SPLIT, split_fractions, split_real  # noqa: F401  (re-exported)
from metis.sota_models.generators.base import BaseGenerator
from metis.sota_models.generators.registry import GeneratorRegistry

logger = logging.getLogger(__name__)

# A family score that is missing (e.g. utility without ``data.target``) is NaN:
# reporting it as 0.0 ranked the generator last on that family.
_NAN = float("nan")


def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_nan_to_none(v) for v in obj]
    return obj


# Default split of the real table per seed (train / holdout / unseen, 60/20/20)
# and the split itself are shared with the calibration (metis.shared.splits).


def _separator(data_cfg: dict[str, Any], prefix: str) -> str:
    for key in (f"{prefix}_separator", "separator"):
        sep = _none_safe(data_cfg.get(key))
        if sep is not None:
            return sep
    return ","


class GeneratorTrainingView:
    """The real table as generators are trained on it, and back.

    Starts from the real table cleaned but not cast and keeps every column
    under its declared type (the four of Phase −1): nominal columns as
    canonical strings, ordinal columns as their raw labels with the schema's
    levels, continuous and discrete ones as floats.  Dates (``continuous``
    with ``datetime: true``) are trained as seconds since the epoch
    (generators cannot model timestamps) and :meth:`restore` turns them back
    into datetimes, so the synthetic table is evaluated with the original schema.
    """

    def __init__(self, real_clean: pd.DataFrame, schema_cfg: dict[str, Any]) -> None:
        from metis.infrastructure.preprocess.caster import SimpleCaster

        self._datetime: list[str] = []
        categorical: list[str] = []
        ordinal: dict[str, list] = {}
        continuous: list[str] = []
        train = pd.DataFrame(index=real_clean.index)
        for col, spec in schema_cfg.items():
            typ = spec if isinstance(spec, str) else spec.get("type", "")
            if typ == "id" or col not in real_clean.columns:
                continue
            s = real_clean[col]
            if typ == "continuous" and isinstance(spec, dict) and spec.get("datetime"):
                ts = pd.to_datetime(s, errors="coerce")
                train[col] = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
                self._datetime.append(col)
                continuous.append(col)
            elif typ == "ordinal":
                train[col] = s.map(SimpleCaster._canonical_str)
                ordinal[col] = [SimpleCaster._canonical_str(v) for v in spec.get("levels", [])]
            elif typ == "nominal":
                train[col] = s.map(SimpleCaster._canonical_str)
                categorical.append(col)
            else:  # continuous, discrete
                train[col] = pd.to_numeric(s, errors="coerce").astype(float)
                continuous.append(col)
        self.train_data = train.reset_index(drop=True)
        self.column_types: dict[str, Any] = {
            "categorical": categorical,
            "ordinal": ordinal,
            "continuous": continuous,
            "id": [],
        }

    def restore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Synthetic (or real-half) table in the real table's representation."""
        cols = [c for c in self.train_data.columns if c in df.columns]
        out = df[cols].copy()
        for col in self._datetime:
            if col in out.columns:
                out[col] = pd.to_datetime(
                    pd.to_numeric(out[col], errors="coerce"), unit="s", errors="coerce"
                )
        return out


class BenchmarkOrchestrator:
    """
    Orchestrator for benchmarking multiple synthetic data generators.

    This class manages the entire benchmarking pipeline:
    - Loading benchmark configuration
    - Instantiating generators
    - Running generators with multiple seeds
    - Evaluating synthetic data with METIS
    - Aggregating results for comparison
    """

    # Delegate to the centralised GeneratorRegistry (Strategy + Registry pattern)
    GENERATOR_REGISTRY = GeneratorRegistry

    def __init__(self, benchmark_config_path: str):
        """
        Initialize benchmark orchestrator.

        Accepts either:
        - **Unified config**: a single YAML with ``data``, ``evaluation``,
          ``calibration``, ``reproducibility``, ``report`` and a ``benchmark``
          section that contains ``generators``, ``output_dir``, etc.
        - **Legacy config** (backward-compatible): a flat YAML with
          ``real_data_path``, ``metis_config_template``, ``generators``, etc.

        Args:
            benchmark_config_path: Path to configuration YAML
        """
        self.benchmark_config_path = benchmark_config_path
        self._raw_config = self._load_raw_config()
        bench_section = self._raw_config.get("benchmark", {})
        self._is_unified = bool(bench_section)
        # A unified config can disable its benchmark via enabled: false
        self._benchmark_enabled = bench_section.get("enabled", True) if self._is_unified else True
        self.benchmark_config = self._build_benchmark_view()
        self.metis_orchestrator = MetisOrchestrator()
        self.results = {}
        # Holds the ephemeral TemporaryDirectory used to stage synthetic
        # CSVs when ``benchmark.persist_artifacts`` is False; cleaned up
        # at the end of :meth:`run_benchmark`.
        self._synth_tmp_ctx: tempfile.TemporaryDirectory | None = None

    # ------------------------------------------------------------------
    # Config loading helpers
    # ------------------------------------------------------------------

    def _load_raw_config(self) -> dict[str, Any]:
        """Load raw YAML without interpretation."""
        config_path = Path(self.benchmark_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.benchmark_config_path}")

        with config_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_benchmark_view(self) -> dict[str, Any]:
        """Return a flat dict with benchmark fields.

        For **unified** configs the fields are derived from the ``data``,
        ``reproducibility`` and ``benchmark`` sections.  For **legacy**
        configs the dict is returned as-is after validation.
        """
        if self._is_unified:
            return self._flatten_unified_config()
        return self._validate_legacy_config(self._raw_config)

    def _flatten_unified_config(self) -> dict[str, Any]:
        """Map a unified config to the flat benchmark view."""
        cfg = self._raw_config
        bench = cfg["benchmark"]
        data = cfg.get("data", {})
        repro = cfg.get("reproducibility", {})

        if "generators" not in bench:
            raise ValueError("Unified config: benchmark.generators is required")
        if "output_dir" not in bench:
            raise ValueError("Unified config: benchmark.output_dir is required")

        return {
            # Data (derived from data section)
            "real_data_path": data.get("real", ""),
            # Same order as the evaluation loader: <prefix>_separator ->
            # separator -> ","; the global ``separator`` was ignored here.
            "real_data_separator": _separator(data, "real"),
            "synthetic_data_separator": _separator(data, "synth"),
            "target_column": data.get("target"),
            # Benchmark-specific
            "generators": bench["generators"],
            "output_dir": bench["output_dir"],
            "sample_ratio": bench.get("sample_ratio", 1.0),
            "seed": repro.get("seed", 42),
            "n_runs": bench.get("n_runs", 3),
            "split": split_fractions(cfg, "benchmark"),
            # Synthetic tables are kept by default: without them the tables of
            # the paper cannot be rebuilt without retraining every generator.
            "persist_artifacts": bench.get("persist_artifacts", True),
            # The template is the unified file itself
            "metis_config_template": self.benchmark_config_path,
            # Statistical test (optional)
            "statistical_test": bench.get("statistical_test"),
        }

    @staticmethod
    def _validate_legacy_config(config: dict[str, Any]) -> dict[str, Any]:
        """Validate a legacy (flat) benchmark config."""
        required = [
            "real_data_path",
            "metis_config_template",
            "generators",
            "output_dir",
        ]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field in benchmark config: {field}")
        return config

    def _get_generator(self, generator_name: str, **kwargs) -> BaseGenerator:
        """
        Instantiate a generator by name.

        Args:
            generator_name: Name of the generator (from registry)
            **kwargs: Additional parameters for generator initialization

        Returns:
            Generator instance

        Raises:
            ValueError: If generator name is unknown
        """
        return self.GENERATOR_REGISTRY.create(generator_name, **kwargs)

    def _extract_column_types(self, metis_config: dict[str, Any]) -> dict[str, list[str]]:
        """Extract column type information from METIS config schema."""
        schema = metis_config.get("data", {}).get("schema", {})
        return extract_column_types(schema)

    def run_benchmark(self) -> dict[str, Any]:
        """
        Run complete benchmark pipeline.

        Returns:
            Dictionary with benchmark results for all generators and seeds

        Raises:
            RuntimeError: If benchmark is disabled (``benchmark.enabled: false``).
        """
        if not self._benchmark_enabled:
            raise RuntimeError(
                "Benchmark is disabled in config (benchmark.enabled: false). "
                "Set benchmark.enabled: true or remove the flag to run."
            )

        print(f"Starting benchmark with config: {self.benchmark_config_path}")

        # Load real data
        real_data_path = self.benchmark_config["real_data_path"]
        print(f"Loading real data from: {real_data_path}")

        separator = self.benchmark_config.get("real_data_separator", ",")
        real_data_raw = load_csv(real_data_path, sep=separator)

        print(f"Loaded real data: {real_data_raw.shape[0]} rows, {real_data_raw.shape[1]} columns")

        # Load METIS config template
        metis_config_path = self.benchmark_config["metis_config_template"]
        with Path(metis_config_path).open(encoding="utf-8") as f:
            metis_config_template = yaml.safe_load(f)

        # Strip the benchmark section — it's not part of the METIS eval config
        metis_config_template.pop("benchmark", None)

        # Filter to only schema columns from METIS config
        schema_columns = list(metis_config_template.get("data", {}).get("schema", {}).keys())
        if schema_columns:
            available_cols = [col for col in schema_columns if col in real_data_raw.columns]
            real_data_raw = real_data_raw[available_cols]
            print(f"Filtered to schema columns: {len(available_cols)} columns")

        # -----------------------------------------------------------
        # METIS preprocessing of the real table: IDs stripped and rows with
        # NaN dropped, but *not* cast.  Generators are trained on this table
        # with the declared types, and their output is evaluated with the
        # original schema, exactly like any synthetic table handed to METIS.
        #
        # Before, generators were trained on the cast table (``real_tf.full``)
        # with every column re-declared as categorical/continuous: an ordinal
        # ``X`` and its rank ``X__ord`` were learnt as two independent
        # columns, datetimes as bare seconds, and the evaluation then ran on
        # a synthetic schema instead of the user's.
        # -----------------------------------------------------------
        preprocessor = DataPreprocessor()
        eval_config = copy.deepcopy(metis_config_template)
        data_cfg = eval_config.get("data", {})
        preprocess_spec = DatasetSpec(
            target=_none_safe(data_cfg.get("target")),
            task_type=_none_safe(data_cfg.get("task_type")),
        )
        reference = preprocessor.prepare_reference(real_data_raw, eval_config, preprocess_spec)
        real_data = reference.real_clean
        train_view = GeneratorTrainingView(real_data, reference.schema_cfg)
        train_data = train_view.train_data
        column_types = train_view.column_types

        print(
            f"Preprocessed: {len(real_data)} rows, "
            f"{len(column_types['categorical'])} cat + {len(column_types['ordinal'])} ord + "
            f"{len(column_types['continuous'])} num columns"
        )

        # -----------------------------------------------------------
        # Pre-compute calibration bounds on the FULL dataset.
        # This guarantees every generator evaluation (including split-
        # half real_data) uses the same bounds and tuned aggregators
        # instead of re-calibrating on a subset of the data.
        # -----------------------------------------------------------
        calibration_bounds_file: str | None = None
        cal_cfg = metis_config_template.get("calibration", {})
        if cal_cfg and cal_cfg.get("enabled", True) and cal_cfg.get("mode") != "default":
            repro_seed = self._raw_config.get("reproducibility", {}).get("seed", 42)
            cal_base_seed = cal_cfg.get("base_seed", repro_seed)
            n_cal_iterations = cal_cfg.get("n_iterations", 10)
            cal_sample_pct = cal_cfg.get("sample_percentage", 100.0)
            cal_sample_size = int(len(real_data) * cal_sample_pct / 100.0)
            # ``real_data`` is the cleaned, uncast table: the same one the
            # evaluation pipeline calibrates on (``ReferenceContext.real_clean``).

            data_section = self._raw_config.get("data", {})
            dataset_name = Path(data_section.get("real", "")).stem or None

            cache_mgr = CacheManager(dataset_name=dataset_name)
            _bounds = cache_mgr.get_or_calibrate(
                real_data=real_data,
                config_path=metis_config_path,
                n_iterations=n_cal_iterations,
                sample_percentage=cal_sample_pct,
                base_seed=cal_base_seed,
                n_jobs=cal_cfg.get("n_jobs", 1),
                tune_aggregators=cal_cfg.get("tune_aggregators", True),
            )

            # Locate the cache file so we can point evaluate_dataframes at it
            data_fp = compute_data_fingerprint(real_data)
            config_fp = compute_config_fingerprint(metis_config_path)
            cache_key = generate_cache_key(
                data_fingerprint=data_fp,
                config_fingerprint=config_fp,
                n_iterations=n_cal_iterations,
                sample_size=cal_sample_size,
                base_seed=cal_base_seed,
            )
            found_path = cache_mgr.find_cache_path(cache_key)
            if found_path:
                calibration_bounds_file = str(found_path)
                print(f"Pre-computed calibration bounds: {found_path.name}")

        # Get generators to run
        generator_configs = self.benchmark_config["generators"]

        # Generate seeds from base seed and n_runs
        base_seed = self.benchmark_config.get("seed", 42)
        n_runs = self.benchmark_config.get("n_runs", 3)
        seeds = [base_seed + i for i in range(n_runs)]

        output_dir = Path(self.benchmark_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------------
        # Temporary directory for persisting synthetic CSVs.
        # Layout / privacy policy:
        #   - persist_artifacts == False: synthetic CSVs go to an
        #     ephemeral tempdir cleaned up at the end of the run; real-data
        #     parts are NEVER written to disk.
        #   - persist_artifacts == True (default): synthetic CSVs are persisted under
        #     ``<output_dir>/synthetic_artifacts/`` (a path the user already
        #     controls). Real-data references remain ephemeral regardless.
        # -----------------------------------------------------------
        _dataset_stem = Path(self.benchmark_config.get("real_data_path", "unknown")).stem
        _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        persist_artifacts = bool(self.benchmark_config.get("persist_artifacts", True))

        if persist_artifacts:
            synth_tmp_dir = (
                Path(self.benchmark_config["output_dir"])
                / "synthetic_artifacts"
                / _timestamp
                / _dataset_stem
            )
            synth_tmp_dir.mkdir(parents=True, exist_ok=True)
            self._synth_tmp_ctx = None
            print(f"Synthetic data will be persisted to: {synth_tmp_dir}")
            logger.info("Persisting synthetic CSVs under controlled output dir: %s", synth_tmp_dir)
        else:
            self._synth_tmp_ctx = tempfile.TemporaryDirectory(
                prefix=f"metis_synth_{_dataset_stem}_"
            )
            synth_tmp_dir = Path(self._synth_tmp_ctx.name)
            logger.info(
                "Synthetic CSVs go to ephemeral tempdir (cleaned at run end). "
                "Set benchmark.persist_artifacts: true to override."
            )

        # Results storage
        all_results = {}

        # Run each generator
        for gen_config in generator_configs:
            gen_name = gen_config["name"]
            gen_params = gen_config.get("params", {})

            print(f"\n{'=' * 60}")
            print(f"Running generator: {gen_name}")
            print(f"{'=' * 60}")

            gen_results = {}

            # Run with multiple seeds
            for seed in seeds:
                print(f"\nSeed: {seed}")

                # Same split for every generator of a seed: each one is
                # trained on ``train_part`` and evaluated against it with the
                # same holdout and the same synthetic size (``real_data`` was
                # evaluated with half the rows of the others).
                task_type = _none_safe(metis_config_template.get("data", {}).get("task_type"))
                target = self.benchmark_config.get("target_column")
                train_part, holdout_part, unseen_part = split_real(
                    train_data,
                    seed,
                    self.benchmark_config.get("split", DEFAULT_SPLIT),
                    stratify_by=target if str(task_type or "").lower() != "regression" else None,
                )

                try:
                    # Add target_column and task_type to params if generator needs it (e.g., SMOTENC)
                    if gen_name == "smotenc" and "target_column" not in gen_params:
                        gen_params_with_target = gen_params.copy()
                        gen_params_with_target["target_column"] = self.benchmark_config.get(
                            "target_column"
                        )
                        task_type = _none_safe(
                            metis_config_template.get("data", {}).get("task_type")
                        )
                        if task_type and "task_type" not in gen_params_with_target:
                            gen_params_with_target["task_type"] = task_type
                    else:
                        gen_params_with_target = gen_params

                    # Instantiate generator with seed
                    generator = self._get_generator(
                        gen_name, random_state=seed, **gen_params_with_target
                    )

                    # ``real_data`` returns rows nobody saw; it is not trained.
                    if hasattr(generator, "set_unseen"):
                        generator.set_unseen(unseen_part)

                    # Fit generator to the training part only
                    print(f"Fitting {generator.name} on {len(train_part)} samples...")
                    generator.fit(
                        real_data=train_part,
                        categorical_columns=column_types["categorical"],
                        ordinal_columns=column_types["ordinal"],
                        continuous_columns=column_types["continuous"],
                    )

                    # Generate synthetic samples: as many as the unseen part
                    # (times sample_ratio, default 1.0).  ``real_data`` returns
                    # the unseen rows, so every generator and the reference are
                    # evaluated at the same size, and METIS compares real and
                    # synthetic at the size of the smaller (``equal_sizes``).
                    sample_ratio = self.benchmark_config.get("sample_ratio", 1.0)
                    n_samples_to_generate = max(1, int(len(unseen_part) * sample_ratio))
                    print(
                        f"Generating {n_samples_to_generate} synthetic samples "
                        f"({sample_ratio:.0%} of the {len(unseen_part)} unseen rows)..."
                    )
                    synthetic_data = generator.generate(n_samples=n_samples_to_generate)

                    # Back to the representation of the real table
                    # (datetimes from seconds), schema columns only.
                    synthetic_data = train_view.restore(synthetic_data)

                    # Persist synthetic data to tmp directory
                    synth_csv_path = synth_tmp_dir / f"{gen_name}_seed{seed}.csv"
                    synthetic_data.to_csv(synth_csv_path, index=False)
                    print(f"Saved synthetic data → {synth_csv_path}")

                    # Every generator: the part it was trained on as the real
                    # table, and the holdout as never-seen reference.  The
                    # real parts are kept in memory only (PII).
                    eval_real = train_view.restore(train_part)
                    eval_holdout = train_view.restore(holdout_part)

                    # Build per-run config
                    run_config = copy.deepcopy(eval_config)
                    run_config["report"] = {
                        "output_dir": str(output_dir / f"{gen_name}_seed{seed}"),
                    }

                    # Point at the pre-computed calibration file so that
                    # evaluate_dataframes loads it directly instead of
                    # re-calibrating (which would use the wrong dataset
                    # size when eval_real is a split-half subset).
                    if calibration_bounds_file:
                        run_config.setdefault("calibration", {})["bounds_file"] = (
                            calibration_bounds_file
                        )

                    # Evaluate in-memory (no temp files)
                    print("Evaluating with METIS...")
                    run_summary = self.metis_orchestrator.evaluate_dataframes(
                        real_data=eval_real,
                        synth_data=synthetic_data,
                        config=run_config,
                        seed=seed,
                        config_path=metis_config_path,
                        holdout=eval_holdout,
                    )

                    gen_results[seed] = {
                        "run_summary": run_summary,
                        "synthetic_path": str(synth_csv_path),
                        "n_train": len(train_part),
                        "n_holdout": len(holdout_part),
                        "privacy_accounting": getattr(generator, "privacy_accounting", None),
                    }

                    print(f"Completed {gen_name} with seed {seed}")

                except MemoryError:
                    error_msg = (
                        f"MEMORY ERROR: {gen_name} with seed {seed} exhausted available RAM. "
                        f"Skipping remaining seeds for this generator."
                    )
                    print(f"\n{'!' * 60}")
                    print(error_msg)
                    print(f"{'!' * 60}")
                    logger.error(error_msg)
                    gen_results[seed] = {"error": f"MemoryError: {error_msg}"}
                    # Force garbage collection to reclaim memory
                    gc.collect()
                    # Skip remaining seeds for this generator — it will OOM again
                    for remaining_seed in seeds[seeds.index(seed) + 1 :]:
                        gen_results[remaining_seed] = {
                            "error": f"Skipped: previous seed {seed} caused MemoryError"
                        }
                    break

                except (KeyboardInterrupt, SystemExit):
                    raise

                except Exception as e:
                    error_msg = f"Failed {gen_name} with seed {seed}: {type(e).__name__}: {e}"
                    print(f"\n{'!' * 60}")
                    print(error_msg)
                    logger.error(error_msg, exc_info=True)
                    print(traceback.format_exc())
                    print(f"{'!' * 60}")
                    gen_results[seed] = {"error": str(e)}
                    # Force garbage collection after failures
                    gc.collect()

            all_results[gen_name] = gen_results

        # Store results
        self.results = all_results

        # Save raw results
        results_path = output_dir / "benchmark_results.json"
        self._save_results(results_path)

        # Release ephemeral tempdir holding synthetic CSVs (no-op when None).
        if getattr(self, "_synth_tmp_ctx", None) is not None:
            try:
                self._synth_tmp_ctx.cleanup()
            except OSError as exc:
                logger.warning("Failed to clean synthetic tempdir: %s", exc)
            self._synth_tmp_ctx = None

        print(f"\n{'=' * 60}")
        print(f"Benchmark complete! Results saved to: {output_dir}")
        print(f"{'=' * 60}")

        return all_results

    def _save_results(self, output_path: Path):
        """
        Save benchmark results to JSON.

        Args:
            output_path: Path to save results
        """
        # Convert RunSummary objects to dictionaries for JSON serialization
        serializable_results = {}

        for gen_name, gen_results in self.results.items():
            serializable_results[gen_name] = {}

            for seed, seed_results in gen_results.items():
                if "error" in seed_results:
                    serializable_results[gen_name][seed] = seed_results
                else:
                    run_summary = seed_results["run_summary"]
                    agg = run_summary.aggregates
                    scores = {
                        "fidelity": agg.get("fidelity_score", _NAN),
                        "utility": agg.get("utility_score", _NAN),
                        "privacy": agg.get("privacy_score", _NAN),
                        "overall": agg.get("composite_score", _NAN),
                    }
                    # Include raw (pre-calibration) scores when available
                    if agg.get("composite_score_calibrated"):
                        scores["fidelity_raw"] = agg.get("fidelity_score_raw")
                        scores["utility_raw"] = agg.get("utility_score_raw")
                        scores["privacy_raw"] = agg.get("privacy_score_raw")
                        scores["calibrated"] = True
                    else:
                        scores["calibrated"] = False

                    artifacts = run_summary.artifacts
                    serializable_results[gen_name][seed] = {
                        "aggregated_scores": scores,
                        "metrics": [
                            {
                                "metric_id": m.id,
                                "value": m.value,
                                "family": m.family,
                                "status": getattr(m, "status", None),
                                "details": self._serializable_details(m.details),
                            }
                            for m in run_summary.results
                        ],
                        "synthetic_path": seed_results.get("synthetic_path"),
                        "n_train": seed_results.get("n_train"),
                        "n_holdout": seed_results.get("n_holdout"),
                        "privacy_accounting": seed_results.get("privacy_accounting"),
                        "synth_rows_dropped_fraction": artifacts.get("synth_rows_dropped_fraction"),
                        "warnings": list(artifacts.get("warnings", [])),
                        "metis_version": artifacts.get("metis_version"),
                        "metis_commit": artifacts.get("metis_commit"),
                    }

        with output_path.open("w", encoding="utf-8") as f:
            # NaN is not JSON: a missing family is written as ``null``.
            json.dump(_nan_to_none(serializable_results), f, indent=2)

    @staticmethod
    def _serializable_details(details: dict | None) -> dict | None:
        """Make metric details JSON-serializable.

        Keeps all scalar / list / dict values and skips anything that
        ``json.dumps`` would reject (e.g. numpy arrays, custom objects).
        """
        if not details:
            return None
        out: dict = {}
        for k, v in details.items():
            if isinstance(v, str | int | float | bool | type(None)):
                out[k] = v
            elif isinstance(v, list | tuple):
                out[k] = list(v)
            elif isinstance(v, dict):
                out[k] = {
                    str(dk): dv
                    for dk, dv in v.items()
                    if isinstance(dv, str | int | float | bool | type(None))
                }
            elif isinstance(v, set):
                out[k] = sorted(v)
            else:
                try:
                    # numpy scalars
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = str(v)
        return out or None

    def get_aggregated_scores(self) -> pd.DataFrame:
        """
        Get aggregated scores for all generators.

        Returns:
            DataFrame with rows=generators, columns=dimensions (fidelity, utility, privacy, overall)
            plus raw (pre-calibration) columns when calibration was applied.
        """
        rows = []

        for gen_name, gen_results in self.results.items():
            fidelity_scores = []
            utility_scores = []
            privacy_scores = []
            overall_scores = []
            fidelity_raw_scores = []
            utility_raw_scores = []
            privacy_raw_scores = []
            has_calibrated = False

            for _seed, seed_results in gen_results.items():
                if "error" not in seed_results:
                    aggregates = seed_results["run_summary"].aggregates
                    fidelity_scores.append(aggregates.get("fidelity_score", _NAN))
                    utility_scores.append(aggregates.get("utility_score", _NAN))
                    privacy_scores.append(aggregates.get("privacy_score", _NAN))
                    overall_scores.append(aggregates.get("composite_score", _NAN))

                    if aggregates.get("composite_score_calibrated"):
                        has_calibrated = True
                        fidelity_raw_scores.append(aggregates.get("fidelity_score_raw", _NAN))
                        utility_raw_scores.append(aggregates.get("utility_score_raw", _NAN))
                        privacy_raw_scores.append(aggregates.get("privacy_score_raw", _NAN))

            n_failed = sum(1 for r in gen_results.values() if "error" in r)
            if n_failed:
                logger.warning(
                    "%s: %d of %d seed(s) failed and are not in its means",
                    gen_name,
                    n_failed,
                    len(gen_results),
                )
            if fidelity_scores:
                row = {
                    "generator": gen_name,
                    # Seeds behind the means: a failed seed used to vanish silently.
                    "n_seeds_ok": len(fidelity_scores),
                    "n_seeds_failed": n_failed,
                    "fidelity_mean": pd.Series(fidelity_scores).mean(),
                    "fidelity_std": pd.Series(fidelity_scores).std(),
                    "utility_mean": pd.Series(utility_scores).mean(),
                    "utility_std": pd.Series(utility_scores).std(),
                    "privacy_mean": pd.Series(privacy_scores).mean(),
                    "privacy_std": pd.Series(privacy_scores).std(),
                    "overall_mean": pd.Series(overall_scores).mean(),
                    "overall_std": pd.Series(overall_scores).std(),
                }
                if has_calibrated and fidelity_raw_scores:
                    row["fidelity_raw_mean"] = pd.Series(fidelity_raw_scores).mean()
                    row["utility_raw_mean"] = pd.Series(utility_raw_scores).mean()
                    row["privacy_raw_mean"] = pd.Series(privacy_raw_scores).mean()
                    row["calibrated"] = True
                rows.append(row)
            else:
                rows.append({"generator": gen_name, "n_seeds_ok": 0, "n_seeds_failed": n_failed})

        return pd.DataFrame(rows)
