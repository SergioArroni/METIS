# Benchmark Mode

Compare multiple synthetic data generators head-to-head with statistical testing.

## Command

```bash
metis evaluate --config config.yaml
```

> Benchmark activates when `benchmark.enabled: true` in the YAML. Uses the same `evaluate` command.

## Requirements

- Real CSV (no synthetic CSV needed — generators produce it)
- `benchmark` section in config with `enabled: true`
- At least 2 generators listed

## How It Works

1. **Pre-calibrate** — Estimates bounds once for the dataset
2. **Generate** — For each generator × each seed: fit on real data → generate N rows
3. **Evaluate** — Compute all configured metrics on each synthetic output
4. **Statistical test** — Friedman test for overall ranking + Nemenyi post-hoc
5. **Report** — Comparative tables, rankings, and critical difference diagrams

## Configuration

```yaml
benchmark:
  enabled: true
  output_dir: "results/benchmark_cardio"
  n_runs: 5
  sample_ratio: 1.0

  generators:
    - name: "real_data"       # Upper bound baseline
      params: {}
    - name: "uniform_noise"   # Lower bound baseline
      params: {}
    - name: "ctgan"
      params: { epochs: 300, batch_size: 500 }
    - name: "tvae"
      params: { epochs: 300 }
    - name: "gaussian_copula"
      params: {}

  statistical_test:
    method: "friedman-nemenyi"
    alpha: 0.05
```

## Available Generators

| Key | Type | Description |
|-----|------|-------------|
| `real_data` | Baseline | Returns real data (upper bound) |
| `uniform_noise` | Baseline | Uniform random noise (lower bound) |
| `bootstrap` | Baseline | Random sampling with replacement |
| `smotenc` | Baseline | SMOTE for mixed-type data |
| `gaussian_copula` | Statistical | Gaussian copula model |
| `bn` | Statistical | Bayesian network |
| `cart` | ML-based | CART-based synthesis |
| `ctgan` | Deep Learning | Conditional GAN |
| `tvae` | Deep Learning | Variational Autoencoder |
| `adsgan` | Deep Learning | Anonymization GAN |
| `dpctgan` | Deep Learning (DP) | Differentially-private CTGAN |

## Output

| File | Contents |
|------|----------|
| `benchmark_results.json` | Raw results per generator × seed |
| `benchmark_comparison.json` | Statistical comparison |
| `benchmark_comparison.md` | Human-readable rankings |
| `scores_raw.csv` | Scores per generator and seed |
| `summary_statistics.csv` | Mean, std, CI per generator |
| `rankings_dimension.csv` | Rankings by dimension |

## Statistical Testing

METIS uses **Friedman test** (non-parametric repeated measures) to detect if generators differ significantly, followed by **Nemenyi post-hoc** to identify which pairs differ.

- α = 0.05 by default
- Results include critical difference values
- Rankings are computed per dimension (Fidelity, Utility, Privacy) and overall
