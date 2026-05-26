# Evaluate Mode

Score a synthetic dataset against the real one across all configured metrics.

## Prerequisites

- A real CSV file
- A synthetic CSV file (same columns)
- A YAML config with `data.synthetic` pointing to the synthetic file

## Command

```bash
metis evaluate --config config.yaml
```

## What Happens Internally

1. **Load** — Reads real + synthetic CSVs
2. **Preprocess** — SimpleCaster transforms heterogeneous types → uniform CAT/NUM views
3. **Validate** — Checks schemas, NaN thresholds, column compatibility
4. **Calibrate** — Estimates bounds (if calibration section present, or uses cached)
5. **Evaluate** — Computes every metric listed in `metrics:`
6. **Aggregate** — Stochastic dominance per family → composite score
7. **Report** — Generates JSON and/or Markdown reports

## Multi-Run Mode

Set `evaluation.n_runs > 1` to run N repetitions with different seeds:

```yaml
evaluation:
  n_runs: 5
```

This produces cross-run statistics (mean, std, confidence intervals) for each metric and score.

## Output

Reports are written to `report.output_dir`:

| File | Contents |
|------|----------|
| `summary.json` | Compact summary with scores |
| `all_metrics.json` | Full details for all metrics |
| `summary.md` | Human-readable report |

## Example

```yaml
data:
  real: "data/real/cardio_train.csv"
  synthetic: "data/synth/cardio_ctgan.csv"
  target: "cardio"
  task_type: "classification"
  schema:
    age: continuous
    gender: categorical
    cholesterol: ordinal
    cardio: categorical

metrics:
  - "fidelity"
  - "utility"
  - "privacy.dataset_based"

calibration:
  n_iterations: 5

evaluation:
  n_runs: 3

report:
  formats: ["json", "md"]
  output_dir: "reports/cardio_ctgan"
```
