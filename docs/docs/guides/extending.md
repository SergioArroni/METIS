# Extending METIS

METIS is designed to be extensible. You can add new metrics and new generators without modifying core code.

## Add a New Metric

### Step 1 — Choose the base class

| Family | Base class | Location |
|--------|-----------|----------|
| Fidelity (numeric) | `NumericColumnMetric` | `metis/infrastructure/metrics/fidelity/fidelity_base.py` |
| Fidelity (categorical) | `CategoricalColumnMetric` | same |
| Fidelity (num↔num) | `NumNumPairMetric` | same |
| Fidelity (num↔cat) | `NumCatPairMetric` | same |
| Fidelity (cat↔cat) | `CatCatPairMetric` | same |
| Privacy | `DatasetPrivacyMetric` | `metis/infrastructure/metrics/privacy/privacy_base.py` |
| Utility | Inherit from utility base | `metis/infrastructure/metrics/utility/` |

### Step 2 — Create the metric file

```python
# metis/infrastructure/metrics/fidelity/marginal/tails/my_metric.py

import pandas as pd
from scipy.stats import some_test

from metis.infrastructure.metrics.registry import register
from ...fidelity_base import NumericColumnMetric


@register("fidelity.my_metric")
class MyMetric(NumericColumnMetric):
    """Brief description of what this metric measures."""

    name: str = "my_metric"
    is_distance: bool = True  # True if lower raw value = better quality

    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """Compute metric for a single numeric column."""
        statistic, _ = some_test(real_col.values, synth_col.values)
        return float(statistic)
```

!!! note
    The `@register("fidelity.my_metric")` decorator is all you need for auto-discovery.

### Step 3 — Import in registry

Add the import in `metis/infrastructure/metrics/registry.py`:

```python
def _register_default_metrics():
    # ... existing imports ...
    from .fidelity.marginal.tails.my_metric import MyMetric
```

### Step 4 — Add to taxonomy

In `metis/domain/taxonomy.py`:

```python
FIDELITY_METRICS = {
    "marginal": {
        "subcategories": {
            "tails": [
                "fidelity.ks",
                # ...
                "fidelity.my_metric",  # ← add here
            ],
        },
    },
}
```

### Step 5 — Use in config

```yaml
metrics:
  - "fidelity.my_metric"        # by exact ID
  - "fidelity.marginal.tails"   # or via shortcut (auto-included)
```

---

## Add a New Generator

### Step 1 — Create the generator class

```python
# metis/sota_models/generators/my_generator.py

import pandas as pd
from .base import BaseGenerator


class MyGenerator(BaseGenerator):
    """Brief description."""

    def __init__(self, name="MyGenerator", my_param=100, random_state=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.my_param = my_param
        self.random_state = random_state

    def fit(self, real_data, categorical_columns=None, ordinal_columns=None,
            continuous_columns=None):
        # Learn distribution
        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted first")
        # Generate synthetic data
        return synth_df
```

### Step 2 — Register

In `metis/sota_models/generators/__init__.py`:

```python
from .registry import GeneratorRegistry
from .my_generator import MyGenerator

GeneratorRegistry.register("my_generator", MyGenerator)
```

### Step 3 — Use in YAML

```yaml
benchmark:
  generators:
    - name: "my_generator"
      params:
        my_param: 200
        random_state: 42
```
