# Calibrate Mode

Estimate empirical bounds for metric normalization.

## Why Calibrate?

Raw metric values (e.g., KS statistic = 0.15) have no universal interpretation. METIS calibrates each metric using:

- **Upper bound** — real vs. real (split-half): the best achievable score
- **Lower bound** — real vs. uniform noise: the worst expected score

After calibration, all metrics are normalized to **[0, 1]** where 1 = best.

## Command

```bash
metis calibrate --config config.yaml

# Override iterations from CLI
metis calibrate --config config.yaml --iterations 20
```

## Requirements

- Only the **real CSV** is needed (no synthetic)
- Set `data.synthetic: "None"` or omit it

## How It Works

1. **Upper bound estimation** — Splits real data in half, computes metric between halves (repeated N times)
2. **Lower bound estimation** — Generates uniform noise matching schema, computes metric (repeated N times)
3. **Aggregator tuning** (optional) — Optimizes aggregation weights with Optuna
4. **Caching** — Persists bounds as JSON with dataset fingerprint

## Caching

Calibration results are cached in `metis/calibrate/cache/` with fingerprint-based validation:

- **Data fingerprint** — SHA-256 of full DataFrame content
- **Config fingerprint** — SHA-256 of relevant config sections (metrics, schema, task_type)
- **Parameter hash** — iterations, sample_size, seed

If data or config changes, the cache is automatically invalidated.

## Configuration

```yaml
calibration:
  n_iterations: 10          # More iterations = tighter bounds
  sample_percentage: 100.0  # Use full dataset
  n_jobs: -1                # Use all CPU cores
  tune_aggregators: true    # Optimize weights with Optuna
```

## Output

Calibration results are stored as JSON files in the cache directory. They're automatically reused by `evaluate` and `benchmark` modes.
