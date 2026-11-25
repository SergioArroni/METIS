# METIS

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**A modular framework for evaluating synthetic tabular data quality, utility, and privacy preservation.**

METIS computes 48 metrics organized across three dimensions — **Fidelity**, **Utility**, and **Privacy** — with empirical bound calibration, comparative generator benchmarking, and statistical testing (Friedman-Nemenyi).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration (YAML)](#configuration-yaml)
- [Running METIS](#running-metis)
  - [Evaluate Mode](#1-evaluate--score-a-synthetic-csv)
  - [Calibrate Mode](#2-calibrate--estimate-metric-bounds)
  - [Benchmark Mode](#3-benchmark--compare-generators)
  - [Python SDK](#programmatic-usage-sdk)
- [Available Metrics (48)](#available-metrics-48)
- [How to Add a New Metric](#how-to-add-a-new-metric)
- [How to Add a Generator to the Benchmark](#how-to-add-a-generator-to-the-benchmark)
- [Development](#development)
- [CI/CD](#cicd)
- [License](#license)

---

## Overview

| Dimension | What it measures | Example metrics |
|-----------|-----------------|-----------------|
| **Fidelity** | Statistical similarity between real and synthetic data | KS, Wasserstein, MMD, Cramér's V |
| **Utility** | Performance on downstream ML tasks | TSTR, TTS, ML Efficiency |
| **Privacy** | Protection against re-identification attacks | DCR, NNAA, MIA, k-Anonymity |

Fidelity and privacy metrics are calibrated **per metric** to **[0, 1]** with empirical anchors: the upper one is real data the generator never saw; the lower one is uniform noise (fidelity) or an exact copy of the training rows (privacy), all measured with a holdout. Utility is not calibrated: its absolute scale `(TSTR − T)/(TTR − T)` already runs from a model without signal (0) to the real baseline (1). The composite score aggregates all three dimensions.

---

## Architecture

```
metis/
├── domain/            # Entities, contracts (Protocol), and metric taxonomy
├── application/       # Orchestrator + injectable pipeline steps
│   └── pipeline/      # Loader → Preprocessor → Validator → Calibrator → Evaluator → Aggregator → Reporter
├── infrastructure/    # Technical implementations
│   ├── io/            # Data loading and schema handling
│   ├── metrics/       # 48 metrics (fidelity/, utility/, privacy/)
│   ├── preprocess/    # SimpleCaster: heterogeneous types → uniform CAT/NUM views
│   └── reporting/     # JSON and Markdown reporters
├── calibrate/         # Bound calibration engine with caching
├── sota_models/       # Benchmark: generators (baselines + SOTA) and statistical comparison
│   ├── generators/    # BaseGenerator + GeneratorRegistry + 15 generators
│   └── comparison/    # BenchmarkOrchestrator, Friedman-Nemenyi, Reporter
├── configs/           # Example YAML configurations per dataset
├── interface/         # CLI (argparse) and public SDK
└── shared/            # Utilities: normalization, distributions, reproducibility
```

---

## Installation

**Requirements:** Python ≥ 3.12

```bash
# From PyPI
pip install metis-val

# With ML extras (catboost, xgboost, lightgbm)
pip install "metis-val[ml]"

# Everything (ML + dev tools)
pip install "metis-val[all]"
```

### From source (development)

```bash
git clone https://github.com/SergioArroni/METIS.git
cd METIS

# Editable install
pip install -e .

# With dev dependencies (pytest, ruff, mypy)
pip install -e ".[dev]"

# With uv (recommended)
uv pip install -e ".[all]"
```

Verify the installation:

```bash
metis version
# or
python -m metis version
```

---

## Configuration (YAML)

METIS uses a **single YAML file** that controls all three modes of operation.
Sections follow a natural flow: *data → metrics → reproducibility → calibration → evaluation → benchmark → reports*.

```yaml
# ── 1. DATA ──────────────────────────────────────────────────────────────

data:
  real: "data/real/cardio_train.csv"
  synthetic: "data/synth/cardio_synth.csv"  # "None" for benchmark-only
  target: "cardio"                           # "None" for fidelity-only
  task_type: "classification"                # classification | regression | "None"
  real_separator: ","
  synth_separator: ","

  # Schema is MANDATORY — defines the semantic type per column
  schema:
    patient_id: id               # Excluded from analysis
    age: continuous
    gender: categorical
    income:                      # Discrete with ranges
      type: discrete
      ranges: [[0, 1000], [1001, 5000], [5001, 20000]]
    education:                   # Ordinal with levels
      type: ordinal
      levels: [primary, secondary, bachelor, master, phd]
    has_insurance: boolean

# ── 2. METRICS ───────────────────────────────────────────────────────────
# You can use exact IDs or hierarchical shortcuts (see taxonomy below).

metrics:
  - "fidelity.global"                              # expands to distances + detection
  - "fidelity.marginal"                            # expands to tails + scales + coverage
  - "fidelity.conditional"                         # expands to num_num + num_cat + cat_cat
  - "utility.tstr"                                 # single metric
  - "utility.tts"
  - "privacy.dataset_based"                        # expands to all dataset-based

# ── 3. REPRODUCIBILITY ──────────────────────────────────────────────────

reproducibility:
  seed: 42

# ── 4. CALIBRATION (optional) ───────────────────────────────────────────

calibration:
  n_iterations: 10         # Iterations per reference (bounds = their mean)
  sample_percentage: 100.0 # % of data to use (split in thirds per iteration)
  n_jobs: 1                # Parallelism

# ── 4b. AGGREGATION (optional) ──────────────────────────────────────────
# Weights on any node of the taxonomy, relative to its siblings (default 1).
# 0 excludes the node. Keys: family, family.category,
# family.category.subcategory, or a metric (its id or its full path).
# Programmatic use: Aggregator().aggregate(results, None, weights={...})

aggregation:
  risk_aversion: 5.0
  weights:
    utility: 0                          # composite without utility
    fidelity.global: 2                  # global counts double inside fidelity
    fidelity.marginal.tails: 3          # tails x3 against scales and coverage
    privacy.dataset_based.attacks: 2
    fidelity.ks: 0                      # = fidelity.marginal.tails.ks

# ── 5. EVALUATION (optional) ────────────────────────────────────────────

evaluation:
  n_runs: 5                # Repetitions for stability (different seeds)

# ── 6. BENCHMARK (optional) ─────────────────────────────────────────────

benchmark:
  enabled: true
  output_dir: "metis/sota_models/results/benchmark_cardio"
  n_runs: 5
  sample_ratio: 1.0

  generators:
    - name: "real_data"
      params: {}
    - name: "uniform_noise"
      params: {}
    - name: "bootstrap"
      params: {}
    - name: "ctgan"
      params: { epochs: 300, batch_size: 500 }
    - name: "tvae"
      params: { epochs: 300, batch_size: 500 }

  statistical_test:
    method: "friedman-nemenyi"
    alpha: 0.05

# ── 7. REPORTS ───────────────────────────────────────────────────────────

report:
  formats: ["json", "md"]
  output_dir: "reports/evaluation"
```

### Supported Column Types

| Type | Description | Output view |
|------|-------------|-------------|
| `continuous` | Continuous numeric values | NUM |
| `discrete` | Discrete numeric with ranges | NUM (normalized) |
| `categorical` | Categorical values | CAT |
| `boolean` | Boolean values | CAT + NUM |
| `ordinal` | Ordered categories | CAT + NUM [0,1] |
| `datetime` | Date/time values | NUM (timestamp) |
| `geospatial` | Lat/lon coordinates | NUM |
| `text` | Free text | CAT (top-k + hash) |
| `code_numeric` | Numeric codes (zip, phone) | CAT |
| `id` | Identifiers | **Excluded** |

### Metric Shortcuts

In the `metrics` section you can use hierarchical shortcuts instead of listing every ID:

| Shortcut | Expands to |
|----------|-----------|
| `"fidelity"` | All 32 fidelity metrics |
| `"fidelity.global"` | `mmd`, `energy_distance` (distances) and `discriminator` (detection) |
| `"fidelity.marginal.tails"` | `ks`, `wasserstein`, `anderson_darling`, `hellinger`, `kde_ise`, `delta_exceedance`, `outliers_coverage` |
| `"fidelity.marginal.scales"` | `delta_mean`, `delta_median`, `delta_iqr`, `delta_mad`, `cohens_d` |
| `"fidelity.marginal.coverage"` | `tvd`, `js`, `kl`, `psi`, `entropy_delta`, `gini_delta` |
| `"fidelity.conditional"` | All conditionals (num↔num, num↔cat, cat↔cat) |
| `"utility"` | All utility metrics |
| `"privacy"` | All privacy metrics |
| `"privacy.dataset_based"` | All except `differential_privacy` |

---

## Running METIS

METIS has **three modes of operation**, all controlled from the same YAML file:

| Mode | CLI command | Required YAML sections | What it does |
|------|------------|----------------------|--------------|
| **Evaluate** | `metis evaluate` | `data` (with synthetic CSV), `metrics`, `report` | Scores a synthetic dataset against the real one |
| **Calibrate** | `metis calibrate` | `data`, `metrics`, `calibration` | Estimates empirical per-metric bounds |
| **Benchmark** | `metis evaluate` | `data`, `metrics`, `benchmark` (enabled: true) | Compares N generators with statistical tests |

### 1. Evaluate — Score a synthetic CSV

**Requirements:** a real CSV and a synthetic CSV. The `data.synthetic` field must point to the synthetic file.

```bash
# CLI
metis evaluate --config metis/configs/config_cardio.yaml

# Equivalent as Python module
python -m metis evaluate --config metis/configs/config_cardio.yaml
```

**What it does internally:**

1. Loads real + synthetic data
2. Preprocesses with SimpleCaster (heterogeneous types → uniform CAT/NUM views)
3. Validates schemas, NaN, column compatibility
4. Calibrates bounds if a `calibration` section exists (or uses existing cache)
5. Evaluates every metric listed in `metrics`
6. Aggregates scores per family → composite score
7. Generates reports in `report.output_dir`

**Output:** `summary.json`, `all_metrics.json`, `summary.md` in the reports directory.

If `evaluation.n_runs > 1`, it runs N repetitions with different seeds and computes cross-run statistics (mean, std, CI).

### 2. Calibrate — Estimate metric bounds

**Requirements:** only the real CSV. The `data.synthetic` field can be `"None"`.

```bash
metis calibrate --config metis/configs/config_cardio.yaml

# Override iterations from CLI
metis calibrate --config metis/configs/config_cardio.yaml --iterations 20
```

**What it does internally:**

Each iteration splits the real table in thirds: *train* (what a generator sees), *holdout* and *unseen*.

1. **Upper anchor** of every fidelity and privacy metric: *unseen* vs *train*, holdout given
2. **Lower anchor**: uniform noise vs *train* (fidelity) and an exact copy of *train* (privacy), holdout given
3. Per metric: anchors = mean of the iterations; a metric whose range is below 5× its noise (the iterations' standard deviations combined) is excluded from the aggregation instead of amplifying noise
4. Utility metrics are not calibrated
5. Persists bounds as JSON with dataset fingerprint; they are reused in evaluate and benchmark via cache

At evaluation each metric is rescaled as `clip((s − lower)/(upper − lower), 0, 1)` before the hierarchical aggregation; `aggregates` keep the raw family scores (`<family>_score_raw`), the calibrated metric values and the excluded metrics. `tune_aggregators` is ignored.

### 3. Benchmark — Compare generators

**Requirements:** real CSV + `benchmark` section with `enabled: true`. No synthetic CSV needed.

```bash
metis evaluate --config metis/configs/config_cardio.yaml
```

> The benchmark activates automatically when `benchmark.enabled: true` in the YAML. It runs with the same `evaluate` command.

**What it does internally:**

1. Pre-calibrates bounds once
2. For each generator × each seed: fit → generate → evaluate with the YAML metrics
3. Persists `benchmark_results.json`
4. Runs statistical test (Friedman + Nemenyi by default)
5. Generates comparative reports

**Output in `benchmark.output_dir`:**

| File | Contents |
|------|----------|
| `benchmark_results.json` | Raw results per generator × seed |
| `benchmark_comparison.json` | Structured statistical comparison |
| `benchmark_comparison.md` | Human-readable report with rankings |
| `scores_raw.csv` | Scores per generator and seed |
| `summary_statistics.csv` | Mean, std, CI per generator and dimension |
| `rankings_dimension.csv` | Rankings per dimension (fidelity, utility, privacy, overall) |

### Programmatic Usage (SDK)

```python
from metis import Evaluator

# From a YAML file
evaluator = Evaluator()
summary = evaluator.run_from_config("metis/configs/config_cardio.yaml")

# From in-memory DataFrames
import pandas as pd

real = pd.read_csv("data/real/cardio_train.csv")
synth = pd.read_csv("data/synth/cardio_synth.csv")
config = {...}  # dict with same structure as YAML

summary = evaluator.evaluate(real, synth, config, seed=42)
```

```python
# Direct shortcut
from metis import evaluate_from_config

summary = evaluate_from_config("metis/configs/config_cardio.yaml")
```

---

## Available Metrics (48)

### Fidelity — Global: Distances and Detection (3)

| ID | Description |
|----|-------------|
| `fidelity.mmd` | Maximum Mean Discrepancy (kernel-based) |
| `fidelity.energy_distance` | Energy distance between distributions |
| `fidelity.discriminator` | Classifier two-sample test: `2·(1 − AUC)` of a CatBoost real-vs-synthetic classifier |

### Fidelity — Marginal: Tails (7)

| ID | Description |
|----|-------------|
| `fidelity.ks` | Kolmogorov-Smirnov test statistic |
| `fidelity.wasserstein` | Wasserstein (earth-mover) distance |
| `fidelity.anderson_darling` | Anderson-Darling test statistic |
| `fidelity.hellinger` | Hellinger distance |
| `fidelity.kde_ise` | Integrated squared error of KDE |
| `fidelity.delta_exceedance` | Exceedance probability delta |
| `fidelity.outliers_coverage` | Coverage of real-data ranges, outliers and density, per column |

### Fidelity — Marginal: Scales (5)

| ID | Description |
|----|-------------|
| `fidelity.delta_mean` | Absolute difference in means |
| `fidelity.delta_median` | Absolute difference in medians |
| `fidelity.delta_iqr` | Absolute difference in IQR |
| `fidelity.delta_mad` | Absolute difference in MAD |
| `fidelity.cohens_d` | Cohen's d effect size |

### Fidelity — Marginal: Coverage (6)

| ID | Description |
|----|-------------|
| `fidelity.tvd` | Total Variation Distance |
| `fidelity.js` | Jensen-Shannon divergence |
| `fidelity.kl` | Kullback-Leibler divergence |
| `fidelity.psi` | Population Stability Index |
| `fidelity.entropy_delta` | Entropy difference |
| `fidelity.gini_delta` | Gini impurity difference |

### Fidelity — Conditional: num↔num (5)

| ID | Description |
|----|-------------|
| `fidelity.pearson` | Pearson correlation delta |
| `fidelity.spearman` | Spearman rank correlation delta |
| `fidelity.mi` | Mutual information delta |
| `fidelity.dcor` | Distance correlation delta |
| `fidelity.correlation_matrix` | Similarity of the whole correlation matrices (Frobenius, spectral, RV) |

### Fidelity — Conditional: num↔cat (3)

| ID | Description |
|----|-------------|
| `fidelity.eta_squared` | Eta-squared (ANOVA effect size) delta |
| `fidelity.point_biserial` | Point-biserial correlation delta |
| `fidelity.kruskal_epsilon` | Kruskal-Wallis epsilon-squared delta |

### Fidelity — Conditional: cat↔cat (3)

| ID | Description |
|----|-------------|
| `fidelity.cramers_v` | Cramér's V delta |
| `fidelity.theils_u` | Theil's U (uncertainty coefficient) delta |
| `fidelity.chi2_stat` | Chi-squared statistic delta |

### Utility (6)

| ID | Description |
|----|-------------|
| `utility.log_loss` | TSTR scored with the log-loss on the same absolute scale (category `probabilistic`) |
| `utility.tstr` | Train-on-Synthetic, Test-on-Real |
| `utility.trts` | Train-on-Real, Test-on-Synthetic |
| `utility.tts` | Train-on-Test-Synthetic |
| `utility.ttrs` | Train-on-Test-Real+Synthetic |
| `utility.ml_efficiency` | ML efficiency: TSTR on an absolute scale, 0 = model without signal, 1 = TTR (CatBoost defaults with early stopping) |

### Privacy (9)

Every dataset-based metric needs a **holdout** (real rows the generator never saw): it is the control of each attack, measured against the same synthetic table. Without one they are not applicable. Membership, linkage and attribute inference are audited in the tail (lower confidence bound of ε from TPR/FPR at low false-positive rates, score `e^{−ε}`): a few percent of copied or near-copied records lowers them clearly, where averages barely move.

Subcategories (the family is their median): `attacks` (mia, inference_attack, record_linkage), `syntactic` (k-anonymity, l-diversity, t-closeness) and `empirical_similarity` (dcr, nnaa). The three audited attacks measure the same signal (correlation ≥ 0.99), so they are one vote; the syntactic models and the average views are the other two.

| ID | Description |
|----|-------------|
| `privacy.dcr` | Holdout DCR test: share of synthetic records closest to a training record vs. its expectation (average view) |
| `privacy.nnaa` | Yale's privacy loss `AA_holdout − AA_train` of the nearest-neighbour adversarial accuracy (average view) |
| `privacy.mia` | Membership inference by relative closeness to the synthetic table, ε-audited |
| `privacy.inference_attack` | Attribute inference from the closest synthetic record (Anonymeter), ε-audited |
| `privacy.record_linkage` | Linkage on the quasi-identifiers by relative closeness, ε-audited |
| `privacy.k_anonymity` | Excess of training records in small synthetic classes over the holdout |
| `privacy.l_diversity` | Excess homogeneity attacks over the holdout |
| `privacy.t_closeness` | Excess skewness attacks over the holdout (EMD of Li et al.) |
| `privacy.differential_privacy` | Differential privacy estimation |

---

## How to Add a New Metric

### Step 1 — Choose the base class

| Family | Metric type | Base class | Location |
|--------|------------|------------|----------|
| Fidelity (numeric column) | Per-column | `NumericColumnMetric` | `metis/infrastructure/metrics/fidelity/fidelity_base.py` |
| Fidelity (categorical column) | Per-column | `CategoricalColumnMetric` | same file |
| Fidelity (num↔num pairs) | Per-pair | `NumNumPairMetric` | same file |
| Fidelity (num↔cat pairs) | Per-pair | `NumCatPairMetric` | same file |
| Fidelity (cat↔cat pairs) | Per-pair | `CatCatPairMetric` | same file |
| Privacy (dataset-based) | Full dataset | `DatasetPrivacyMetric` / `EmpiricalSimilarityMetric` | `metis/infrastructure/metrics/privacy/privacy_base.py` |
| Utility (ML efficiency) | Full dataset | Inherit from base in `metis/infrastructure/metrics/utility/ml_efficiency/base.py` | utility/ |

### Step 2 — Create the file

Create a `.py` file inside the corresponding subfolder. Example for a fidelity marginal (tails) metric:

```python
# metis/infrastructure/metrics/fidelity/marginal/tails/my_metric.py

import pandas as pd
from scipy.stats import ks_2samp  # or whatever library you need

from metis.infrastructure.metrics.registry import register
from ...fidelity_base import NumericColumnMetric


@register("fidelity.my_metric")
class MyMetric(NumericColumnMetric):
    """Description of what the metric measures."""

    name: str = "my_metric"
    is_distance: bool = True  # True if lower = better; False if higher = better

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """Compute the metric value for a single numeric column."""
        # Your logic here
        return float(some_statistic)
```

> **The `@register("fidelity.my_metric")` decorator is all you need** for the metric to enter the global registry.

### Step 3 — Import in `_register_default_metrics`

Add the import in `metis/infrastructure/metrics/registry.py`, inside the `_register_default_metrics()` function:

```python
def _register_default_metrics():
    # ... existing imports ...

    # Your new metric
    from .fidelity.marginal.tails.my_metric import MyMetric
```

### Step 4 — Add to the taxonomy

In `metis/domain/taxonomy.py`, add your metric ID in the corresponding list:

```python
FIDELITY_METRICS = {
    "marginal": {
        "subcategories": {
            "tails": [
                "fidelity.ks",
                "fidelity.wasserstein",
                # ...
                "fidelity.my_metric",  # ← here
            ],
        },
    },
}
```

### Step 5 — Use in configuration

```yaml
metrics:
  - "fidelity.my_metric"       # by exact ID
  - "fidelity.marginal.tails"  # or automatically included via shortcut
```

### Real Example — KS Metric (complete)

```python
# metis/infrastructure/metrics/fidelity/marginal/tails/ks.py

import pandas as pd
from scipy.stats import ks_2samp

from metis.infrastructure.metrics.registry import register
from ...fidelity_base import NumericColumnMetric as TailMetric


@register("fidelity.ks")
class KSMetric(TailMetric):
    """Kolmogorov-Smirnov two-sample test."""

    name: str = "ks"
    is_distance: bool = True  # Lower KS = better

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        statistic, _ = ks_2samp(real_col.values, synth_col.values)
        return float(statistic)
```

---

## How to Add a Generator to the Benchmark

### Step 1 — Create the generator class

Inherit from `BaseGenerator` and implement `fit()` and `generate()`:

```python
# metis/sota_models/generators/my_generator.py

import pandas as pd
from .base import BaseGenerator


class MyGenerator(BaseGenerator):
    """Brief description of the generator."""

    def __init__(
        self,
        name: str = "MyGenerator",
        my_param: int = 100,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.my_param = my_param
        self.random_state = random_state

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Learn the distribution of the real data."""
        # Your training logic
        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate n_samples rows of synthetic data."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating")

        # Your generation logic
        synth = ...
        return synth
```

### Step 2 — Register in the GeneratorRegistry

In `metis/sota_models/generators/__init__.py`, add the registration:

```python
from .registry import GeneratorRegistry
from .my_generator import MyGenerator

GeneratorRegistry.register("my_generator", MyGenerator)
```

### Step 3 — Use in YAML configuration

```yaml
benchmark:
  enabled: true
  generators:
    - name: "my_generator"
      params:
        my_param: 200
        random_state: 42
```

> The `params` are passed as `**kwargs` to the generator constructor.

### Available Generators (15)

| YAML key | Class | Type |
|----------|-------|------|
| `real_data` | `RealDataGenerator` | Baseline (upper bound) |
| `uniform_noise` | `UniformNoiseGenerator` | Baseline (lower bound) |
| `bootstrap` | `RandomSamplingGenerator` | Baseline |
| `smotenc` | `SMOTENCGenerator` | Baseline |
| `delete_zero` | `DeleteImputeZeroGenerator` | Baseline |
| `delete_mean` | `DeleteImputeMeanGenerator` | Baseline |
| `gaussian_copula` | `GaussianCopulaGenerator` | Baseline |
| `bn` / `bayesian_network` | `BayesianNetworkGenerator` | SOTA |
| `cart` | `CARTGenerator` | SOTA |
| `ctgan` | `CTGANGenerator` | SOTA |
| `tvae` | `TVAEGenerator` | SOTA |
| `adsgan` | `ADSGANGenerator` | SOTA |
| `dpctgan` | `DPCTGANGenerator` | SOTA (DP) |

---

## Development

### Tests

```bash
# Unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=metis --cov-report=html

# Integration tests
pytest tests/integration/ -v
```

### Linting and Formatting

```bash
# Check style
ruff check metis/

# Auto-fix
ruff check metis/ --fix

# Format
ruff format metis/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## CI/CD

GitHub Actions with three pipelines:

- **Lint**: Ruff (linting + formatting)
- **Test**: Unit tests with coverage
- **Version bump** automatic per branch:
  - `main` → major
  - `staging` → minor
  - `development` → patch

---

## License

MIT — see [LICENSE](LICENSE).
