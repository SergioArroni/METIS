"""
Shared utilities for metric results, distributions and configuration.

This module contains common functionality used across all metric types:
- Result data structures (ColumnMetricResult)
- Distribution utilities
- Normalization helpers
- Reproducibility utilities (seed management)
- Schema utilities (column type parsing)
"""

from .distributions import align_distributions, get_distribution
from .reproducibility import configure_deterministic_mode, get_seed_for_run, set_global_seed
from .results import ColumnMetricResult
from .schema_utils import extract_column_types, filter_schema_columns

__all__ = [
    # Distribution utilities
    "get_distribution",
    "align_distributions",
    # Result structures
    "ColumnMetricResult",
    # Reproducibility utilities
    "set_global_seed",
    "get_seed_for_run",
    "configure_deterministic_mode",
    # Schema utilities
    "extract_column_types",
    "filter_schema_columns",
]
