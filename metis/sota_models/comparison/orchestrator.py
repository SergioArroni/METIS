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
from metis.sota_models.generators.base import BaseGenerator
from metis.sota_models.generators.registry import GeneratorRegistry

logger = logging.getLogger(__name__)


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
            "real_data_separator": data.get("real_separator", ","),
            "synthetic_data_separator": data.get("synth_separator", ","),
            "target_column": data.get("target"),
            # Benchmark-specific
            "generators": bench["generators"],
            "output_dir": bench["output_dir"],
            "sample_ratio": bench.get("sample_ratio", 1.0),
            "seed": repro.get("seed", 42),
            "n_runs": bench.get("n_runs", 3),
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
        # METIS preprocessing: transform raw data → clean cat/num
        # This ensures generators receive data with only categorical
        # (string) and continuous (float) columns, matching what METIS
        # metrics actually evaluate.  Fixes generators like SMOTE that
        # cannot handle raw text / datetime / geospatial dtypes.
        # -----------------------------------------------------------
        # NOTE: SMOTENC natively handles categorical features, but still
        # needs clean data without mixed/object dtypes.
        preprocessor = DataPreprocessor()
        eval_config_for_preprocess = copy.deepcopy(metis_config_template)
        data_cfg = eval_config_for_preprocess.get("data", {})
        preprocess_spec = DatasetSpec(
            target=_none_safe(data_cfg.get("target")),
            task_type=_none_safe(data_cfg.get("task_type")),
        )
        real_tf, _ = preprocessor.preprocess(
            real_data_raw,
            real_data_raw.copy(),
            eval_config_for_preprocess,
            preprocess_spec,
        )
        real_data = real_tf.full  # clean DataFrame: cat (str) + num (float)

        # Build clean column types from the preprocessed columns
        clean_cat_cols = list(real_tf.cat.columns)
        clean_num_cols = list(real_tf.num.columns)
        column_types = {
            "categorical": clean_cat_cols,
            "ordinal": {},
            "continuous": clean_num_cols,
            "id": [],
        }

        # Build clean schema for evaluation: every column is now either
        # "categorical" or "continuous" — no text/datetime/geospatial.
        clean_schema: dict[str, str] = {}
        for col in clean_cat_cols:
            clean_schema[col] = "categorical"
        for col in clean_num_cols:
            clean_schema[col] = "continuous"

        # Build eval config with the clean schema
        eval_config = copy.deepcopy(metis_config_template)
        eval_config["data"]["schema"] = clean_schema

        print(
            f"Preprocessed: {len(real_data)} rows, "
            f"{len(clean_cat_cols)} cat + {len(clean_num_cols)} num columns"
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
        #   - persist_artifacts == False (default): synthetic CSVs go to an
        #     ephemeral tempdir cleaned up at the end of the run; real-data
        #     reference halves are NEVER written to disk.
        #   - persist_artifacts == True: synthetic CSVs are persisted under
        #     ``<output_dir>/synthetic_artifacts/`` (a path the user already
        #     controls). Real-data references remain ephemeral regardless.
        # -----------------------------------------------------------
        _dataset_stem = Path(self.benchmark_config.get("real_data_path", "unknown")).stem
        _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        persist_artifacts = bool(self.benchmark_config.get("persist_artifacts", False))

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

                    # Fit generator to real data
                    print(f"Fitting {generator.name} on {len(real_data)} samples...")
                    generator.fit(
                        real_data=real_data,
                        categorical_columns=column_types["categorical"],
                        ordinal_columns=column_types["ordinal"],
                        continuous_columns=column_types["continuous"],
                    )

                    # Generate synthetic samples
                    # sample_ratio is a proportion of the preprocessed dataset size
                    # (default 1.0 = 100%).  This avoids hard-coding an absolute
                    # count that could mismatch after preprocessing drops rows.
                    sample_ratio = self.benchmark_config.get("sample_ratio", 1.0)
                    n_samples_to_generate = max(1, int(len(real_data) * sample_ratio))
                    print(
                        f"Generating {n_samples_to_generate} synthetic samples "
                        f"({sample_ratio:.0%} of {len(real_data)})..."
                    )
                    synthetic_data = generator.generate(n_samples=n_samples_to_generate)

                    # Filter to only include clean schema columns
                    clean_cols = list(clean_schema.keys())
                    available_synth = [col for col in clean_cols if col in synthetic_data.columns]
                    synthetic_data = synthetic_data[available_synth]

                    # Persist synthetic data to tmp directory
                    synth_csv_path = synth_tmp_dir / f"{gen_name}_seed{seed}.csv"
                    synthetic_data.to_csv(synth_csv_path, index=False)
                    print(f"Saved synthetic data → {synth_csv_path}")

                    # If the generator provides a custom real reference
                    # (e.g. RealDataGenerator uses split-half to be consistent
                    # with the calibration upper bound), use it instead of the
                    # full dataset.
                    eval_real = real_data
                    real_ref = getattr(generator, "real_reference", None)
                    if real_ref is not None:
                        available_real_ref = [col for col in clean_cols if col in real_ref.columns]
                        eval_real = real_ref[available_real_ref]
                        print(
                            f"Using generator's split-half real reference "
                            f"({len(eval_real)} rows) for evaluation"
                        )
                        # SECURITY: real-data half is PII-bearing for some
                        # datasets (e.g. cardio, RRHH). It is intentionally
                        # NOT written to disk; metrics consume it in-memory.

                    # Build per-run config
                    run_config = copy.deepcopy(eval_config)
                    run_config["report"] = {
                        "output_dir": str(output_dir / f"{gen_name}_seed{seed}"),
                    }
                    if "evaluation" not in run_config:
                        run_config["evaluation"] = {}
                    run_config["evaluation"]["dataset_sizes"] = {
                        "n_synthetic": len(synthetic_data),
                        "n_real": len(eval_real),
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
                    )

                    gen_results[seed] = {
                        "run_summary": run_summary,
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
                        "fidelity": agg.get("fidelity_score", 0.0),
                        "utility": agg.get("utility_score", 0.0),
                        "privacy": agg.get("privacy_score", 0.0),
                        "overall": agg.get("composite_score", 0.0),
                    }
                    # Include raw (pre-calibration) scores when available
                    if agg.get("composite_score_calibrated"):
                        scores["fidelity_raw"] = agg.get("fidelity_score_raw")
                        scores["utility_raw"] = agg.get("utility_score_raw")
                        scores["privacy_raw"] = agg.get("privacy_score_raw")
                        scores["calibrated"] = True
                    else:
                        scores["calibrated"] = False

                    serializable_results[gen_name][seed] = {
                        "aggregated_scores": scores,
                        "metrics": [
                            {
                                "metric_id": m.id,
                                "value": m.value,
                                "family": m.family,
                                "details": self._serializable_details(m.details),
                            }
                            for m in run_summary.results
                        ],
                    }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)

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
                    fidelity_scores.append(aggregates.get("fidelity_score", 0.0))
                    utility_scores.append(aggregates.get("utility_score", 0.0))
                    privacy_scores.append(aggregates.get("privacy_score", 0.0))
                    overall_scores.append(aggregates.get("composite_score", 0.0))

                    if aggregates.get("composite_score_calibrated"):
                        has_calibrated = True
                        fidelity_raw_scores.append(aggregates.get("fidelity_score_raw", 0.0))
                        utility_raw_scores.append(aggregates.get("utility_score_raw", 0.0))
                        privacy_raw_scores.append(aggregates.get("privacy_score_raw", 0.0))

            if fidelity_scores:
                row = {
                    "generator": gen_name,
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

        return pd.DataFrame(rows)
