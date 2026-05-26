<div align="center">

# METIS

### A Modular Framework for Evaluating Synthetic Tabular Data Quality, Utility & Privacy

**48 metrics · 3 dimensions · Empirical calibration · Statistical benchmarking**

### [Documentation & Website →](https://SergioArroni.github.io/METIS/)

[![PyPI version](https://img.shields.io/pypi/v/metis-val.svg)](https://pypi.org/project/metis-val/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/sonar/coverage/SergioArroni_METIS?server=https%3A%2F%2Fsonarcloud.io)](https://sonarcloud.io/project/overview?id=SergioArroni_METIS)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=SergioArroni_METIS&metric=alert_status)](https://sonarcloud.io/project/overview?id=SergioArroni_METIS)

[![Windows](https://img.shields.io/badge/Windows-supported-blue.svg)](#installation)
[![macOS](https://img.shields.io/badge/macOS-supported-blue.svg)](#installation)
[![Linux](https://img.shields.io/badge/Linux-supported-blue.svg)](#installation)

</div>

---

## Why METIS?

Evaluating synthetic tabular data requires more than a single metric. Different downstream tasks demand different quality guarantees — statistical fidelity, ML utility preservation, or privacy protection. Existing tools measure one dimension at a time, leaving practitioners to assemble ad-hoc pipelines with incompatible scales.

**METIS solves this** by providing a unified, calibrated evaluation framework:

| Dimension | What it measures | Metrics |
|-----------|-----------------|---------|
| **Fidelity** | Statistical similarity between real and synthetic distributions | 26 metrics: KS, Wasserstein, MMD, Cramér's V, MI, … |
| **Utility** | Performance preservation on downstream ML tasks | 5 metrics: TSTR, TRTS, TTS, TTRS, ML Efficiency |
| **Privacy** | Protection against re-identification and inference attacks | 9 metrics: DCR, NNAA, MIA, k-Anonymity, l-Diversity, … |

All metrics are normalized to **[0, 1]** using empirical bounds (real-vs-real as upper, real-vs-noise as lower), making scores **comparable across datasets and generators**.

---

## Key Features

| | |
|---|---|
| **48 Calibrated Metrics** | Fidelity, Utility, and Privacy — all normalized to [0,1] with empirical bounds |
| **Three Operating Modes** | Evaluate a single synthetic CSV, calibrate bounds, or benchmark N generators head-to-head |
| **Empirical Calibration** | Real-vs-real (upper bound) and real-vs-noise (lower bound) via split-half iterations |
| **Statistical Benchmarking** | Compare generators with Friedman-Nemenyi tests and automatic rankings |
| **Stochastic Dominance Aggregation** | Aggregate per-column scores using first-order stochastic dominance |
| **Multi-Run Stability** | N repetitions with different seeds → mean, std, CI per metric |
| **Extensible Registry** | Add a metric with one `@register` decorator — auto-discovered by the taxonomy |
| **YAML-Driven** | Single config file controls data, metrics, calibration, evaluation, and benchmark |
| **CLI + Python SDK** | Use from the command line or embed in notebooks and pipelines |
| **100% Local** | No data leaves your machine. No API keys. No external services |

---

## Installation

**Requirements:** Python ≥ 3.12

```bash
pip install metis-val
```

### From source (development)

```bash
git clone https://github.com/SergioArroni/METIS.git
cd METIS
pip install -e ".[dev]"
```

Verify:

```bash
metis version
# METIS 1.0.0
```

---

## Usage (end-to-end example)

The repository ships with **4 ready-to-use datasets** in `data/real/` and their pre-configured YAML files in `metis/configs/`. To evaluate your own synthetic data you just need to generate the CSV and point the config at it.

### Option A — Use an existing config (fastest)

```bash
pip install metis-val

# Edit the config to point synthetic to your CSV
# metis/configs/config_cardio.yaml → set synthetic: "data/synth/my_cardio_synth.csv"

metis evaluate --config metis/configs/config_cardio.yaml
```

### Option B — From Python

```python
import pandas as pd
from metis import Evaluator

# 1. Load data (4 example datasets included: cardio, telco, rrhh, airbnb)
real = pd.read_csv("data/real/cardio_train.csv")
synth = pd.read_csv("data/synth/my_cardio_ctgan.csv")  # your synthetic CSV

# 2. Define configuration (or load from YAML with evaluate_from_config)
config = {
    "data": {
        "target": "cardio",
        "task_type": "classification",
        "schema": {
            "age": "continuous",
            "gender": "categorical",
            "height": "continuous",
            "weight": "continuous",
            "ap_hi": "continuous",
            "ap_lo": "continuous",
            "cholesterol": "categorical",
            "gluc": "categorical",
            "smoke": "categorical",
            "alco": "categorical",
            "active": "categorical",
            "cardio": "categorical",
        },
    },
    "metrics": [
        "fidelity",            # all 26 fidelity metrics
        "utility.tstr",        # Train-on-Synthetic, Test-on-Real
        "privacy.dcr",         # Distance to Closest Record
        "privacy.nnaa",        # Nearest Neighbor Adversarial Accuracy
    ],
    "calibration": {
        "n_iterations": 5,
    },
    "evaluation": {
        "n_runs": 3,
    },
    "reproducibility": {
        "seed": 42,
    },
    "report": {
        "formats": ["json", "md"],
        "output_dir": "reports/",
    },
}

# 3. Run
evaluator = Evaluator()
summary = evaluator.evaluate(real, synth, config)

# 4. Inspect results
print(f"Composite score: {summary.aggregates['composite_score']:.3f}")
print(f"Fidelity:        {summary.aggregates['fidelity_score']:.3f}")
print(f"Utility:         {summary.aggregates['utility_score']:.3f}")
print(f"Privacy:         {summary.aggregates['privacy_score']:.3f}")

for result in summary.results[:5]:
    print(f"  {result.id}: {result.value:.3f}")
```

### Option C — One-liner from YAML

```python
from metis import evaluate_from_config

summary = evaluate_from_config("metis/configs/config_cardio.yaml")
```

### Available example datasets

| Dataset | Config | Rows | Task |
|---------|--------|------|------|
| `data/real/cardio_train.csv` | `metis/configs/config_cardio.yaml` | 70k | Classification |
| `data/real/telco.csv` | `metis/configs/config_telco.yaml` | 7k | Classification |
| `data/real/rrhh.csv` | `metis/configs/config_rrhh.yaml` | 1.5k | Classification |
| `data/real/airbnb_barcelona.csv` | `metis/configs/config_airbnb.yaml` | 16k | Regression |

To use your own generator, just add its synthetic output as a CSV and set `synthetic: "path/to/your_synth.csv"` in the config.

---

## Quick Start

### 1. Write a config

```yaml
data:
  real: "data/real/cardio_train.csv"
  synthetic: "data/synth/cardio_synth.csv"
  target: "cardio"
  task_type: "classification"
  schema:
    age: continuous
    gender: categorical
    cholesterol: ordinal
    cardio: categorical

metrics:
  - "fidelity"
  - "utility.tstr"
  - "privacy.dataset_based"

calibration:
  n_iterations: 5

report:
  formats: ["json", "md"]
  output_dir: "reports/"
```

### 2. Run evaluation

```bash
metis evaluate --config config.yaml
```

### 3. Check results

```
reports/
├── summary.json       # Machine-readable results
├── all_metrics.json   # Per-metric details
└── summary.md         # Human-readable report
```

---

## Operating Modes

| Mode | Command | What it does |
|------|---------|--------------|
| **Evaluate** | `metis evaluate -c config.yaml` | Score a synthetic dataset against the real one |
| **Calibrate** | `metis calibrate -c config.yaml` | Estimate empirical per-metric bounds |
| **Benchmark** | `metis evaluate -c config.yaml` | Compare N generators with statistical tests (when `benchmark.enabled: true`) |

---

## Benchmark Results

Tested across **7 real-world datasets** comparing 13 generators (baselines + SOTA). Each generator is evaluated on all 48 metrics across 5 seeds. Rankings determined by Friedman test + Nemenyi post-hoc (α=0.05).

| Dataset | Domain | Rows | Cols | Best Generator | Composite Score |
|---------|--------|------|------|----------------|-----------------|
| **Cardio** | Healthcare | 70k | 12 | TVAE | 0.82 |
| **Telco** | Telecom | 7k | 21 | CTGAN | 0.79 |
| **Airbnb** | Real Estate | 16k | 16 | Gaussian Copula | 0.76 |
| **RRHH** | HR Analytics | 1.5k | 35 | CTGAN | 0.81 |
| **HiperAM (clean)** | Manufacturing | 50 | 8 | Bayesian Network | 0.74 |
| **HiperAM (dupl.)** | Manufacturing | 50 | 8 | Bootstrap | 0.71 |
| **AMMaraging** | Materials | 50 | 12 | CART | 0.73 |

<details>
<summary><strong>Available Generators (13)</strong></summary>

| Key | Type | Description |
|-----|------|-------------|
| `real_data` | Baseline (upper bound) | Returns real data (calibration ceiling) |
| `uniform_noise` | Baseline (lower bound) | Uniform random noise (calibration floor) |
| `bootstrap` | Baseline | Random sampling with replacement |
| `smotenc` | Baseline | SMOTE for mixed-type data |
| `delete_zero` | Baseline | Delete + impute with zeros |
| `delete_mean` | Baseline | Delete + impute with means |
| `gaussian_copula` | Statistical | Gaussian copula model |
| `bn` | Statistical | Bayesian network |
| `cart` | ML-based | Classification and Regression Trees |
| `ctgan` | Deep Learning | Conditional GAN for tabular data |
| `tvae` | Deep Learning | Variational Autoencoder for tabular data |
| `adsgan` | Deep Learning | Anonymization through data synthesis GAN |
| `dpctgan` | Deep Learning (DP) | Differentially-private CTGAN |

</details>

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         metis evaluate -c config.yaml                  │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                           Orchestrator                                 │
│                                                                       │
│   Load → Preprocess → Validate → Calibrate → Evaluate → Aggregate    │
│                                                        → Report       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌───────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│     Fidelity      │ │     Utility     │ │     Privacy     │
│   26 metrics      │ │    5 metrics    │ │    9 metrics    │
│                   │ │                 │ │                 │
│ Global (4)        │ │ TSTR, TRTS      │ │ DCR, NNAA, MIA  │
│ Marginal (17)     │ │ TTS, TTRS       │ │ k-Anon, l-Div   │
│ Conditional (10)  │ │ ML Efficiency   │ │ t-Close, DP     │
└───────────────────┘ └─────────────────┘ └─────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     Stochastic Dominance Aggregation                   │
│                                                                       │
│   Per-column scores → First-order SD → Family score → Composite       │
└───────────────────────────────────────────────────────────────────────┘
```

```
metis/
├── domain/            # Entities, contracts (Protocol), taxonomy
├── application/       # Orchestrator + pipeline steps
│   └── pipeline/      # Loader → Preprocessor → Validator → Calibrator → Evaluator → Aggregator → Reporter
├── infrastructure/    # Technical implementations
│   ├── io/            # Data loading and schema handling
│   ├── metrics/       # 48 metrics (fidelity/, utility/, privacy/)
│   ├── preprocess/    # SimpleCaster: heterogeneous types → uniform views
│   └── reporting/     # JSON and Markdown reporters
├── calibrate/         # Bound calibration engine with caching
├── sota_models/       # Benchmark: generators + statistical comparison
├── interface/         # CLI (argparse) and public SDK
└── shared/            # Utilities: normalization, distributions, reproducibility
```

---

## Available Metrics (48)

<details>
<summary><strong>Fidelity — Global (4)</strong></summary>

| ID | Description |
|----|-------------|
| `fidelity.correlation_matrix` | Frobenius distance between correlation matrices |
| `fidelity.mmd` | Maximum Mean Discrepancy (kernel-based) |
| `fidelity.energy_distance` | Energy distance between distributions |
| `fidelity.outliers_coverage` | Coverage of real-data outlier regions |

</details>

<details>
<summary><strong>Fidelity — Marginal: Tails (6)</strong></summary>

| ID | Description |
|----|-------------|
| `fidelity.ks` | Kolmogorov-Smirnov test statistic |
| `fidelity.wasserstein` | Wasserstein (earth-mover) distance |
| `fidelity.anderson_darling` | Anderson-Darling test statistic |
| `fidelity.hellinger` | Hellinger distance |
| `fidelity.kde_ise` | Integrated squared error of KDE |
| `fidelity.delta_exceedance` | Exceedance probability delta |

</details>

<details>
<summary><strong>Fidelity — Marginal: Location & Scale (5)</strong></summary>

| ID | Description |
|----|-------------|
| `fidelity.delta_mean` | Absolute difference in means |
| `fidelity.delta_median` | Absolute difference in medians |
| `fidelity.delta_iqr` | Absolute difference in IQR |
| `fidelity.delta_mad` | Absolute difference in MAD |
| `fidelity.cohens_d` | Cohen's d effect size |

</details>

<details>
<summary><strong>Fidelity — Marginal: Coverage (6)</strong></summary>

| ID | Description |
|----|-------------|
| `fidelity.tvd` | Total Variation Distance |
| `fidelity.js` | Jensen-Shannon divergence |
| `fidelity.kl` | Kullback-Leibler divergence |
| `fidelity.psi` | Population Stability Index |
| `fidelity.entropy_delta` | Entropy difference |
| `fidelity.gini_delta` | Gini impurity difference |

</details>

<details>
<summary><strong>Fidelity — Conditional (10)</strong></summary>

| ID | Description |
|----|-------------|
| `fidelity.pearson` | Pearson correlation delta |
| `fidelity.spearman` | Spearman rank correlation delta |
| `fidelity.mi` | Mutual information delta |
| `fidelity.dcor` | Distance correlation delta |
| `fidelity.eta_squared` | Eta-squared (ANOVA) delta |
| `fidelity.point_biserial` | Point-biserial correlation delta |
| `fidelity.kruskal_epsilon` | Kruskal-Wallis epsilon² delta |
| `fidelity.cramers_v` | Cramér's V delta |
| `fidelity.theils_u` | Theil's U delta |
| `fidelity.chi2_stat` | Chi-squared statistic delta |

</details>

<details>
<summary><strong>Utility (5)</strong></summary>

| ID | Description |
|----|-------------|
| `utility.tstr` | Train-on-Synthetic, Test-on-Real |
| `utility.trts` | Train-on-Real, Test-on-Synthetic |
| `utility.tts` | Train-on-Test-Synthetic |
| `utility.ttrs` | Train-on-Test-Real+Synthetic |
| `utility.ml_efficiency` | Aggregated ML efficiency score |

</details>

<details>
<summary><strong>Privacy (9)</strong></summary>

| ID | Description |
|----|-------------|
| `privacy.dcr` | Distance to Closest Record |
| `privacy.nnaa` | Nearest Neighbor Adversarial Accuracy |
| `privacy.mia` | Membership Inference Attack |
| `privacy.inference_attack` | Attribute inference attack |
| `privacy.record_linkage` | Record linkage attack |
| `privacy.k_anonymity` | k-Anonymity preservation |
| `privacy.l_diversity` | l-Diversity preservation |
| `privacy.t_closeness` | t-Closeness preservation |
| `privacy.differential_privacy` | Differential privacy estimation |

</details>

---

## Configuration

METIS uses a **single YAML file** controlling all modes. Sections follow a natural flow:

```yaml
# ── DATA ─────────────────────────────────────────────────────────────────
data:
  real: "data/real/dataset.csv"
  synthetic: "data/synth/dataset_synth.csv"   # "None" for calibrate/benchmark-only
  target: "label_column"                       # "None" for fidelity-only
  task_type: "classification"                  # classification | regression | "None"
  schema:
    patient_id: id              # Excluded from analysis
    age: continuous
    gender: categorical
    income:
      type: discrete
      ranges: [[0, 1000], [1001, 5000], [5001, 20000]]
    education:
      type: ordinal
      levels: [primary, secondary, bachelor, master, phd]

# ── METRICS ──────────────────────────────────────────────────────────────
metrics:
  - "fidelity"                   # All 26 fidelity metrics
  - "utility.tstr"               # Single metric by ID
  - "privacy.dataset_based"      # Hierarchical shortcut

# ── CALIBRATION ──────────────────────────────────────────────────────────
calibration:
  n_iterations: 5
  sample_percentage: 100.0
  tune_aggregators: true

# ── EVALUATION ───────────────────────────────────────────────────────────
evaluation:
  n_runs: 5                      # Multi-seed stability

# ── BENCHMARK ────────────────────────────────────────────────────────────
benchmark:
  enabled: true
  generators:
    - name: "ctgan"
      params: { epochs: 300 }
    - name: "tvae"
      params: { epochs: 300 }
  statistical_test:
    method: "friedman-nemenyi"
    alpha: 0.05

# ── REPORTS ──────────────────────────────────────────────────────────────
report:
  formats: ["json", "md"]
  output_dir: "reports/"
```

<details>
<summary><strong>Supported column types</strong></summary>

| Type | Description | Internal view |
|------|-------------|---------------|
| `continuous` | Continuous numeric values | NUM |
| `discrete` | Discrete numeric with ranges | NUM (normalized) |
| `categorical` | Categorical values | CAT |
| `boolean` | Boolean values | CAT + NUM |
| `ordinal` | Ordered categories with levels | CAT + NUM [0,1] |
| `datetime` | Date/time values | NUM (timestamp) |
| `geospatial` | Lat/lon coordinates | NUM |
| `text` | Free text | CAT (top-k + hash) |
| `code_numeric` | Numeric codes (zip, phone) | CAT |
| `id` | Identifiers | **Excluded** |

</details>

<details>
<summary><strong>Metric shortcuts</strong></summary>

| Shortcut | Expands to |
|----------|-----------|
| `"fidelity"` | All 26 fidelity metrics |
| `"fidelity.global"` | correlation_matrix, mmd, energy_distance, outliers_coverage |
| `"fidelity.marginal.tails"` | ks, wasserstein, anderson_darling, hellinger, kde_ise, delta_exceedance |
| `"fidelity.marginal.scales"` | delta_mean, delta_median, delta_iqr, delta_mad, cohens_d |
| `"fidelity.marginal.coverage"` | tvd, js, kl, psi, entropy_delta, gini_delta |
| `"fidelity.conditional"` | All conditionals (num↔num, num↔cat, cat↔cat) |
| `"utility"` | All 5 utility metrics |
| `"privacy"` | All 9 privacy metrics |
| `"privacy.dataset_based"` | All except differential_privacy |

</details>

---

## Python SDK

```python
from metis import Evaluator, evaluate_from_config

# From a YAML file
summary = evaluate_from_config("metis/configs/config_cardio.yaml")

# Programmatic usage
import pandas as pd

evaluator = Evaluator()
real = pd.read_csv("data/real/cardio_train.csv")
synth = pd.read_csv("data/synth/cardio_synth.csv")

config = {
    "data": {"target": "cardio", "task_type": "classification",
             "schema": {"age": "continuous", "gender": "categorical"}},
    "metrics": ["fidelity.ks", "privacy.dcr"],
    "reproducibility": {"seed": 42},
}

summary = evaluator.evaluate(real, synth, config)
print(summary.aggregates["composite_score"])  # 0.78
```

---

## CLI Reference

```bash
metis evaluate --config config.yaml      # Run evaluation pipeline
metis calibrate --config config.yaml     # Estimate empirical bounds
metis calibrate -c config.yaml -n 20    # Override iterations
metis version                            # Show version
```

---

## How to Extend

<details>
<summary><strong>Add a new metric</strong></summary>

```python
# metis/infrastructure/metrics/fidelity/marginal/tails/my_metric.py

import pandas as pd
from metis.infrastructure.metrics.registry import register
from ...fidelity_base import NumericColumnMetric


@register("fidelity.my_metric")
class MyMetric(NumericColumnMetric):
    """Description of what the metric measures."""

    name: str = "my_metric"
    is_distance: bool = True  # True if lower = better

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        return float(some_statistic)
```

Then add the import in `metis/infrastructure/metrics/registry.py` and the ID in `metis/domain/taxonomy.py`.

</details>

<details>
<summary><strong>Add a generator to the benchmark</strong></summary>

```python
# metis/sota_models/generators/my_generator.py

import pandas as pd
from .base import BaseGenerator


class MyGenerator(BaseGenerator):
    def fit(self, real_data, categorical_columns=None, **kwargs):
        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        return synth_df
```

Register in `metis/sota_models/generators/__init__.py`:
```python
GeneratorRegistry.register("my_generator", MyGenerator)
```

Use in YAML:
```yaml
benchmark:
  generators:
    - name: "my_generator"
      params: { my_param: 42 }
```

</details>

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v

# Coverage report
pytest tests/ --cov=metis --cov-report=html --cov-report=xml

# Linting
ruff check metis tests
ruff format metis tests

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

---

## CI/CD

| Pipeline | Trigger | What it does |
|----------|---------|--------------|
| **Lint** | Push / PR | Ruff linting + formatting check |
| **Test** | Push / PR | Unit + integration tests with coverage |
| **SonarCloud** | Push / PR | Static analysis + quality gate |
| **Docs** | Push to main | Build & deploy MkDocs to GitHub Pages |
| **Publish** | Release tag | Build + upload to PyPI |
| **Version bump** | Push to main/staging/dev | Auto-increment version |

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Built for researchers and practitioners evaluating synthetic tabular data**

[Report Bug](https://github.com/SergioArroni/METIS/issues) · [Request Feature](https://github.com/SergioArroni/METIS/issues) · [Documentation](https://SergioArroni.github.io/METIS/)

</div>
