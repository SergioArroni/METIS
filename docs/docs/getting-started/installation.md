# Installation

## Requirements

- Python ≥ 3.12

## From PyPI

```bash
# Core package
pip install metis-val

# With ML extras (CatBoost, XGBoost, LightGBM)
pip install "metis-val[ml]"

# Everything (ML + dev tools)
pip install "metis-val[all]"
```

## From Source

```bash
git clone https://github.com/SergioArroni/METIS.git
cd METIS

# With uv (recommended)
uv pip install -e ".[all]"

# With pip
pip install -e ".[dev]"
```

## Verify Installation

```bash
metis version
# METIS 0.1.0
```

## Optional Dependencies

| Extra | Includes | Use case |
|-------|----------|----------|
| `ml` | CatBoost, XGBoost, LightGBM | ML efficiency metrics with multiple models |
| `dev` | pytest, ruff, mypy, pre-commit | Development and testing |
| `all` | `ml` + `dev` | Full installation |
