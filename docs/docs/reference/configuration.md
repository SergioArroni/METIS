# YAML Configuration Reference

Complete reference for all YAML configuration options.

## `data` Section

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `real` | string | Yes | Path to real CSV file |
| `synthetic` | string | No | Path to synthetic CSV (or `"None"`) |
| `target` | string | No | Target column name (or `"None"`) |
| `task_type` | string | No | `classification`, `regression`, or `"None"` |
| `real_separator` | string | No | CSV separator for real file (default: `,`) |
| `synth_separator` | string | No | CSV separator for synthetic file (default: `,`) |
| `schema` | dict | Yes | Column type definitions |

## `metrics` Section

List of metric IDs or shortcuts:

```yaml
metrics:
  - "fidelity.ks"            # Single metric
  - "fidelity.marginal"      # Shortcut (all marginal)
  - "privacy"                # All privacy metrics
```

## `reproducibility` Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `seed` | int | 42 | Base random seed |

## `calibration` Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_iterations` | int | 10 | Split-half calibration iterations |
| `sample_percentage` | float | 100.0 | % of dataset to use |
| `n_jobs` | int | 1 | Parallel jobs (-1 = all cores) |
| `tune_aggregators` | bool | true | Optimize aggregation weights |

## `evaluation` Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_runs` | int | 1 | Number of repetitions (multi-seed) |

## `benchmark` Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | false | Enable benchmark mode |
| `output_dir` | string | — | Output directory for results |
| `n_runs` | int | 5 | Seeds per generator |
| `sample_ratio` | float | 1.0 | Fraction of data to use |
| `generators` | list | — | Generator definitions |
| `statistical_test.method` | string | `friedman-nemenyi` | Test method |
| `statistical_test.alpha` | float | 0.05 | Significance level |

### Generator definition

```yaml
generators:
  - name: "ctgan"            # Registry key
    params:                   # Passed as kwargs to constructor
      epochs: 300
      batch_size: 500
```

## `report` Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `formats` | list | `["json"]` | Output formats: `json`, `md` |
| `output_dir` | string | `reports/` | Output directory |
