# Python SDK

## Overview

METIS provides two entry points for programmatic usage:

- `Evaluator` class — full control with method calls
- `evaluate_from_config()` — one-liner convenience function

## Quick Usage

```python
from metis import Evaluator, evaluate_from_config

# One-liner
summary = evaluate_from_config("config.yaml")

# Full control
evaluator = Evaluator()
summary = evaluator.run_from_config("config.yaml")
```

## `Evaluator` Class

```python
from metis import Evaluator

evaluator = Evaluator()
```

### `run_from_config(config_path: str) -> RunSummary`

Run the full pipeline from a YAML config file.

```python
summary = evaluator.run_from_config("metis/configs/config_cardio.yaml")
print(summary.aggregates["composite_score"])
```

### `evaluate(real, synth, config, seed=42) -> RunSummary`

Evaluate pre-loaded DataFrames.

```python
import pandas as pd

real = pd.read_csv("data/real/cardio.csv")
synth = pd.read_csv("data/synth/cardio_synth.csv")

config = {
    "data": {
        "target": "cardio",
        "task_type": "classification",
        "schema": {
            "age": "continuous",
            "gender": "categorical",
            "cardio": "categorical",
        },
    },
    "metrics": ["fidelity.ks", "fidelity.wasserstein", "privacy.dcr"],
    "reproducibility": {"seed": 42},
    "evaluation": {"n_runs": 1},
    "report": {"output_dir": "reports/", "formats": ["json"]},
}

summary = evaluator.evaluate(real, synth, config, seed=42)
```

## `evaluate_from_config(config_path: str) -> RunSummary`

Convenience function equivalent to `Evaluator().run_from_config(path)`.

```python
from metis import evaluate_from_config

summary = evaluate_from_config("config.yaml")
```

## `RunSummary` Object

The result of any evaluation:

```python
summary.plan           # EvalPlan: metric_ids, seed, cv_splits
summary.results        # list[MetricResult]: per-metric results
summary.aggregates     # dict: fidelity_score, utility_score, privacy_score, composite_score
summary.artifacts      # dict: metadata (schema_applied, seeds_used, etc.)
```

### `MetricResult`

```python
for result in summary.results:
    print(f"{result.id}: {result.value:.3f} ({result.family})")
    # fidelity.ks: 0.850 (fidelity)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Metric identifier (e.g., `fidelity.ks`) |
| `value` | `float` | Normalized score [0, 1] |
| `family` | `str` | Dimension: fidelity, utility, or privacy |
| `details` | `dict` | Per-column or additional details |
| `purpose_tags` | `set[str]` | Semantic tags |
