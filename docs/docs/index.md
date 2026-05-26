# METIS

**A Modular Framework for Evaluating Synthetic Tabular Data Quality, Utility & Privacy**

---

## What is METIS?

METIS is a comprehensive evaluation framework that computes **48 metrics** organized across three dimensions — **Fidelity**, **Utility**, and **Privacy** — to assess synthetic tabular data quality.

Unlike single-metric tools, METIS provides:

- **Empirical calibration** — normalizes all metrics to [0,1] using data-driven bounds
- **Statistical benchmarking** — compares generators with Friedman-Nemenyi tests
- **Stochastic dominance aggregation** — produces a single composite score

## Quick Install

```bash
pip install metis-val
```

## Quick Example

```bash
metis evaluate --config config.yaml
```

```python
from metis import evaluate_from_config

summary = evaluate_from_config("config.yaml")
print(summary.aggregates["composite_score"])
```

## Three Dimensions

| Dimension | Metrics | What it measures |
|-----------|---------|-----------------|
| **Fidelity** | 26 | Statistical similarity to real data |
| **Utility** | 5 | ML task performance preservation |
| **Privacy** | 9 | Protection against attacks |

## Get Started

- [Installation](getting-started/installation.md) — install from PyPI or source
- [Quick Start](getting-started/quickstart.md) — run your first evaluation in 5 minutes
- [Configuration](getting-started/configuration.md) — YAML reference
