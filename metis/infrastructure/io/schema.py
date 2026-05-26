"""Schema validation and alignment utilities."""

import logging
from typing import Any

import pandas as pd

from ...domain.contracts import TypeSchema
from ...domain.entities import DatasetSpec, TransformedData
from ...domain.errors import SchemaError
from ..preprocess.caster import SimpleCaster

logger = logging.getLogger(__name__)


def align_schema(
    real_data: pd.DataFrame,
    synth_data: pd.DataFrame,
    spec: DatasetSpec,
    type_schema: TypeSchema | None = None,
) -> tuple[TransformedData, TransformedData, pd.Series | None]:
    """
    Align schemas of real and synthetic datasets according to specification.

    When a TypeSchema is provided (required), applies SimpleCaster to transform
    heterogeneous data into uniform CAT/NUM DataFrames.

    Args:
        real_data: Real dataset
        synth_data: Synthetic dataset
        spec: Dataset specification with target, dtypes, constraints
        type_schema: TypeSchema defining semantic types for each column (REQUIRED)

    Returns:
        tuple of (real_transformed, synth_transformed, target_column)
        Where *_transformed are TransformedData entities with cat, num, full DataFrames

    Raises:
        SchemaError: If alignment fails
        TypeCastingError: If type casting fails
    """
    # Schema is now required
    if type_schema is None:
        raise SchemaError(
            "TypeSchema is required. Please define 'schema:' section in config "
            "with type definitions for all columns."
        )

    # Validate schema columns exist in both datasets
    SimpleCaster.validate_schema_columns(type_schema, real_data, synth_data)

    # Extract target column before transformation if specified
    target_column = None
    if spec.target:
        for t in spec.target_list:
            if t not in real_data.columns:
                raise SchemaError(f"Target column '{t}' not found in real data")
            if t not in synth_data.columns:
                raise SchemaError(f"Target column '{t}' not found in synthetic data")

        # Keep the first target column for backward compatibility
        target_column = real_data[spec.target_list[0]].copy()

    # Apply SimpleCaster transformation
    caster = SimpleCaster(type_schema)
    caster.fit(real_data)

    # Transform both datasets using fitted caster
    real_transformed, real_warnings = caster.transform_to_entity(real_data, "real")
    synth_transformed, synth_warnings = caster.transform_to_entity(synth_data, "synth")

    # Log warnings for missing categories
    for warning in real_warnings + synth_warnings:
        logger.warning(warning)

    return real_transformed, synth_transformed, target_column


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """
    Infer semantic column types (numeric, categorical, datetime, etc.).

    Args:
        df: Input DataFrame

    Returns:
        Dictionary mapping column names to semantic types
    """
    column_types = {}

    for col in df.columns:
        dtype = df[col].dtype

        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            column_types[col] = "datetime"

        # Check for numeric types
        elif pd.api.types.is_numeric_dtype(dtype):
            # Distinguish between continuous and discrete
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())

            if unique_count <= min(10, total_count * 0.05):
                column_types[col] = "categorical_numeric"
            else:
                column_types[col] = "numeric"

        # Check for boolean
        elif pd.api.types.is_bool_dtype(dtype):
            column_types[col] = "boolean"

        # Object/string types
        else:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())

            # High cardinality suggests text/ID
            if unique_count > total_count * 0.5:
                column_types[col] = "text"
            else:
                column_types[col] = "categorical"

    return column_types


def validate_schema_compatibility(
    real_data: pd.DataFrame, synth_data: pd.DataFrame
) -> dict[str, Any]:
    """
    Validate that two datasets have compatible schemas.

    Args:
        real_data: Real dataset
        synth_data: Synthetic dataset

    Returns:
        Validation report with compatibility status and issues
    """
    report = {"compatible": True, "issues": [], "warnings": []}

    real_cols = set(real_data.columns)
    synth_cols = set(synth_data.columns)

    # Check column presence
    missing_in_synth = real_cols - synth_cols
    if missing_in_synth:
        report["compatible"] = False
        report["issues"].append(f"Columns missing in synthetic: {missing_in_synth}")

    extra_in_synth = synth_cols - real_cols
    if extra_in_synth:
        report["warnings"].append(f"Extra columns in synthetic: {extra_in_synth}")

    # Check common columns
    common_cols = real_cols.intersection(synth_cols)

    for col in common_cols:
        real_dtype = real_data[col].dtype
        synth_dtype = synth_data[col].dtype

        # type compatibility check
        if not _are_compatible_types(real_dtype, synth_dtype):
            report["issues"].append(f"Incompatible types for {col}: {real_dtype} vs {synth_dtype}")
            report["compatible"] = False

        # Value range checks for numeric columns
        if pd.api.types.is_numeric_dtype(real_dtype) and pd.api.types.is_numeric_dtype(synth_dtype):
            real_range = (real_data[col].min(), real_data[col].max())
            synth_range = (synth_data[col].min(), synth_data[col].max())

            # Check if synthetic range is much larger than real range
            if (synth_range[1] - synth_range[0]) > 2 * (real_range[1] - real_range[0]):
                report["warnings"].append(f"Synthetic range much larger for {col}")

    return report


def standardize_missing_values(df: pd.DataFrame, strategy: str = "pandas") -> pd.DataFrame:
    """
    Standardize missing value representation across dataset.

    Args:
        df: Input DataFrame
        strategy: Missing value strategy ("pandas", "zero", "mode")

    Returns:
        DataFrame with standardized missing values

    Raises:
        ValueError: if ``strategy`` is not registered.
    """
    df_clean = df.copy()
    impute = _MISSING_VALUE_STRATEGIES.get(strategy)
    if impute is None:
        raise ValueError(
            f"Unknown missing-value strategy {strategy!r}. "
            f"Registered: {sorted(_MISSING_VALUE_STRATEGIES)}"
        )
    return impute(df_clean)


def _impute_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Identity strategy: keep pandas NA representation."""
    return df


def _impute_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN with 0 for numeric, ``"missing"`` for non-numeric."""
    for col in df.columns:
        fill = 0 if pd.api.types.is_numeric_dtype(df[col]) else "missing"
        df[col] = df[col].fillna(fill)
    return df


def _impute_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN with the column's mode (most frequent value)."""
    for col in df.columns:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])
    return df


# Strategy registry. Add new strategies here without touching the
# dispatching function above (OCP-friendly).
_MISSING_VALUE_STRATEGIES: dict[str, callable] = {
    "pandas": _impute_pandas,
    "zero": _impute_zero,
    "mode": _impute_mode,
}


def register_missing_value_strategy(name: str, fn) -> None:
    """Register a custom missing-value imputation strategy.

    Useful for downstream extensions / plugins that need to add a new
    ``data.missing_values`` value without forking ``schema.py``.
    """
    _MISSING_VALUE_STRATEGIES[name] = fn


def _are_compatible_types(dtype1, dtype2) -> bool:
    """Check if two pandas dtypes are compatible."""
    import numpy as np

    # Convert to numpy dtypes
    np_dtype1 = np.dtype(dtype1)
    np_dtype2 = np.dtype(dtype2)

    # Exact match
    if np_dtype1 == np_dtype2:
        return True

    # Both numeric
    if np.issubdtype(np_dtype1, np.number) and np.issubdtype(np_dtype2, np.number):
        return True

    # Both string-like
    if (np_dtype1 == np.object_ or np_dtype1.kind in ["U", "S"]) and (
        np_dtype2 == np.object_ or np_dtype2.kind in ["U", "S"]
    ):
        return True

    # Both datetime
    return np.issubdtype(np_dtype1, np.datetime64) and np.issubdtype(np_dtype2, np.datetime64)
