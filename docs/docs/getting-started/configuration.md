# Configuration

METIS uses a single YAML file to control all modes of operation.

## Full Schema

```yaml
# ── DATA ─────────────────────────────────────────────────────────────────
data:
  real: "path/to/real.csv"
  synthetic: "path/to/synth.csv"     # "None" for calibrate/benchmark-only
  target: "label_column"              # "None" for fidelity-only evaluation
  task_type: "classification"         # classification | regression | "None"
  real_separator: ","
  synth_separator: ","

  schema:
    column_name: type_spec            # See column types below

# ── METRICS ──────────────────────────────────────────────────────────────
metrics:
  - "fidelity"                        # Shortcut: all fidelity metrics
  - "utility.tstr"                    # Single metric by ID
  - "privacy.dataset_based"           # Shortcut: subset

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────
reproducibility:
  seed: 42

# ── CALIBRATION ──────────────────────────────────────────────────────────
calibration:
  n_iterations: 5                     # Split-half iterations
  sample_percentage: 100.0            # % of data to use
  n_jobs: 1                           # Parallelism (-1 for all cores)
  tune_aggregators: true              # Optimize weights with Optuna

# ── EVALUATION ───────────────────────────────────────────────────────────
evaluation:
  n_runs: 5                           # Multi-seed repetitions

# ── BENCHMARK ────────────────────────────────────────────────────────────
benchmark:
  enabled: true
  output_dir: "results/benchmark"
  n_runs: 5
  sample_ratio: 1.0

  generators:
    - name: "ctgan"
      params: { epochs: 300, batch_size: 500 }
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

## Column Types

| Type | Syntax | Internal View |
|------|--------|---------------|
| `continuous` | `age: continuous` | NUM |
| `discrete` | `income: {type: discrete, ranges: [[0,1000],[1001,5000]]}` | NUM (normalized) |
| `categorical` | `gender: categorical` | CAT |
| `boolean` | `flag: boolean` | CAT + NUM |
| `ordinal` | `edu: {type: ordinal, levels: [low, mid, high]}` | CAT + NUM [0,1] |
| `datetime` | `created: datetime` | NUM (timestamp) |
| `geospatial` | `lat: geospatial` | NUM |
| `text` | `desc: text` | CAT (top-k) |
| `code_numeric` | `zip: code_numeric` | CAT |
| `id` | `patient_id: id` | **Excluded** |

## Metric Shortcuts

| Shortcut | Expands to |
|----------|-----------|
| `"fidelity"` | All 26 fidelity metrics |
| `"fidelity.global"` | correlation_matrix, mmd, energy_distance, outliers_coverage |
| `"fidelity.marginal.tails"` | ks, wasserstein, anderson_darling, hellinger, kde_ise, delta_exceedance |
| `"fidelity.marginal.scales"` | delta_mean, delta_median, delta_iqr, delta_mad, cohens_d |
| `"fidelity.marginal.coverage"` | tvd, js, kl, psi, entropy_delta, gini_delta |
| `"fidelity.conditional"` | All conditional metrics |
| `"utility"` | All 5 utility metrics |
| `"privacy"` | All 9 privacy metrics |
| `"privacy.dataset_based"` | All except differential_privacy |
