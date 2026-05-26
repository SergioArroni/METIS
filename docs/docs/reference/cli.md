# CLI Reference

## Commands

### `metis evaluate`

Run the evaluation pipeline.

```bash
metis evaluate --config <path>
metis evaluate -c <path>
```

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to YAML configuration file (required) |

### `metis calibrate`

Run bound calibration.

```bash
metis calibrate --config <path> [--iterations N]
metis calibrate -c <path> -n 20
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to YAML configuration file (required) | — |
| `--iterations` | `-n` | Override `calibration.n_iterations` | from YAML |

### `metis version`

Show the installed version.

```bash
metis version
# METIS 0.1.0
```

## Examples

```bash
# Evaluate a synthetic dataset
metis evaluate -c metis/configs/config_cardio.yaml

# Calibrate bounds with 20 iterations
metis calibrate -c metis/configs/config_cardio.yaml -n 20

# Benchmark (via evaluate with benchmark.enabled: true)
metis evaluate -c metis/configs/config_cardio.yaml
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (printed to stderr) |
