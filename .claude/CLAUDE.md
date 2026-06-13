# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

METIS is a Python framework for evaluating synthetic tabular data across three dimensions — fidelity (26 metrics), utility (5), privacy (9) — with empirical calibration that normalizes all scores to [0, 1]. Published on PyPI as `metis-val`. Requires Python >=3.12,<3.13.

## Commands

The project uses `uv` (CI does too); plain pip also works.

```bash
# Install for development
uv pip install -e ".[dev,ml]"

# Tests
uv run pytest tests/unit/ -v                  # unit tests
uv run pytest tests/integration/ -v           # integration tests (continue-on-error in CI)
uv run pytest tests/unit/test_caster.py -v    # single file
uv run pytest tests/unit/test_caster.py::TestClass::test_name  # single test
uv run pytest tests/ --cov=metis --cov-report=xml             # coverage (SonarCloud reads coverage.xml)

# Lint / format (line length 100, target py312)
uvx ruff check metis tests scripts
uvx ruff format metis tests scripts

# Run the tool
metis evaluate --config metis/configs/config_cardio.yaml
metis calibrate --config config.yaml [-n N]   # estimate empirical bounds only
metis version
```

Pre-commit hooks (ruff lint/format/isort + a non-blocking pytest run via `scripts/pytest_check.py`) are configured in `.pre-commit-config.yaml`. CI auto-bumps `version.py` on pushes to main/staging/development via `scripts/bump_version.py` — never edit the version manually.

## Architecture

Layered, Protocol-based (contracts in `metis/domain/contracts.py`, no ABC inheritance — structural typing):

- **`metis/domain/`** — pure contracts and frozen dataclasses, no implementations. `entities.py` (DatasetSpec, EvalPlan, MetricResult, RunSummary, TransformedData), `taxonomy.py` (the metric ID hierarchy and shorthand expansion, e.g. `"fidelity"` → all 26 fidelity IDs), `errors.py`.
- **`metis/application/`** — `orchestrator.py` chains seven pipeline steps in `application/pipeline/`: **Loader → Preprocessor → Validator → Calibrator → Evaluator → Aggregator → Reporter**. Multi-run mode (`evaluation.n_runs > 1`) repeats the pipeline with seeds `base_seed + i`, writes per-run JSON to `__runs__/run_{i}.json`, and merges mean/std/CI stats into the final summary.
- **`metis/infrastructure/`** — implementations: `io/` (CSV loading, schema), `metrics/` (the 48 metrics by family), `preprocess/` (SimpleCaster), `reporting/` (JSON + Markdown reporters), `runtime/` (config, logging, StatsStore in-memory cache).
- **`metis/calibrate/`** — calibration engine. `MetricCalibrator` runs UpperBoundStrategy (real-vs-real) and LowerBoundStrategy (real-vs-noise) over N split-half iterations, and `AggregatorTuner` picks per-family aggregation functions. `CacheManager` fingerprints data (1000-row sample hash) + config and caches bounds as JSON under `metis/calibrate/cache/`.
- **`metis/sota_models/`** — benchmark mode: 13 generators (each extends `generators/base.py`, registered in `generators/__init__.py` via `GeneratorRegistry.register`) plus Friedman-Nemenyi statistical comparison. Activated by `benchmark.enabled: true` in config.
- **`metis/interface/`** — `cli.py` (argparse entry point, `metis` script) and `sdk.py` (`Evaluator`, `evaluate_from_config` — the only public exports in `metis/__init__.py`).
- **`metis/shared/`** — aggregation function registry (mean, median, FSD, SSD, …), reproducibility helpers, normalization.

### Key concepts that span multiple files

- **Metric registry**: metrics are classes decorated with `@register("family.metric_id")` (`metis/infrastructure/metrics/registry.py`). Registration only happens when the module is imported, so every metric must also be imported in `_register_default_metrics()` in registry.py, and its ID must exist in `metis/domain/taxonomy.py` for config shorthand expansion to find it. To add a metric: create the class (extend the appropriate base in `metrics/`, e.g. `NumericColumnMetric`), decorate, import in registry, add to taxonomy.
- **SimpleCaster views** (`metis/infrastructure/preprocess/caster.py`): heterogeneous column types (continuous, categorical, ordinal, datetime, text, geospatial, …) are cast into uniform views on `TransformedData`: `.num` (floats), `.cat` (strings), `.full`. `id` columns are dropped entirely. The caster fits on real data and applies the same mapping to synthetic. Metrics declare which view they consume.
- **Two-level aggregation** (`metis/infrastructure/metrics/aggregation/stochastic_dominance.py`): per-column scores roll up via first-order stochastic dominance at the lower levels and second-order (risk-averse) at category→family→composite. Calibration bounds are injected into the Aggregator to linearly rescale family scores to [0, 1].
- **Seeds** (`metis/shared/reproducibility.py`): priority is `evaluation.seed` → `reproducibility.seed` → 42. `set_global_seed()` seeds random/numpy/torch and PYTHONHASHSEED.
- **Single YAML config** drives everything (data + schema, metrics list, calibration, evaluation, benchmark, report). Example configs for the four bundled datasets live in `metis/configs/`; the datasets themselves in `data/real/`.

## Tests

`tests/unit/` and `tests/integration/`; shared fixtures (sample dataframes, configs, metric results) in `tests/conftest.py`. Some tests use `hypothesis` (it's in the dev/ci extras). Long-running tests are marked `pytest.mark.slow`.
