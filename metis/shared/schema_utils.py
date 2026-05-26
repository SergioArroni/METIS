"""Shared utilities for parsing METIS schema configuration."""

from typing import Any

from metis.infrastructure.runtime.logging import get_logger

# Column types that are excluded from distributional evaluation.
EXCLUDED_SCHEMA_TYPES = frozenset({"id"})
_logger = get_logger(__name__)


def extract_column_types(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract column type lists from a METIS schema dict.

    Parses the ``data.schema`` section of a METIS config and classifies
    each column into categorical, ordinal, continuous, or excluded (id).

    Args:
        schema: The ``config["data"]["schema"]`` dictionary mapping
                column names to type specs (str or dict).

    Returns:
        Dictionary with keys ``categorical`` (list), ``ordinal`` (dict
        mapping column to levels), ``continuous`` (list), ``id`` (list).
    """
    categorical: list[str] = []
    ordinal: dict[str, list] = {}
    continuous: list[str] = []
    id_columns: list[str] = []

    for col_name, col_spec in schema.items():
        if isinstance(col_spec, dict):
            col_type = col_spec.get("type", "continuous")
            levels = col_spec.get("levels", [])
        else:
            col_type = col_spec
            levels = []

        if col_type in EXCLUDED_SCHEMA_TYPES:
            id_columns.append(col_name)
        elif col_type in ("categorical", "boolean", "text", "code_numeric"):
            categorical.append(col_name)
        elif col_type == "ordinal":
            ordinal[col_name] = levels
        elif col_type in ("continuous", "discrete", "datetime", "geospatial"):
            continuous.append(col_name)
        else:
            # Unknown type — default to continuous
            continuous.append(col_name)

    _logger.info(
        "Extracted column types: %d categorical, %d ordinal, %d continuous, %d id",
        len(categorical),
        len(ordinal),
        len(continuous),
        len(id_columns),
    )

    return {
        "categorical": categorical,
        "ordinal": ordinal,
        "continuous": continuous,
        "id": id_columns,
    }


def filter_schema_columns(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *schema* with excluded column types removed.

    Args:
        schema: The ``config["data"]["schema"]`` dictionary.

    Returns:
        Filtered schema dict containing only evaluable columns.
    """
    filtered = {}
    for col, spec in schema.items():
        col_type = spec if isinstance(spec, str) else spec.get("type", "")
        if col_type not in EXCLUDED_SCHEMA_TYPES:
            filtered[col] = spec
    _logger.info(
        "Filtered schema columns: %d total, %d excluded, %d evaluable",
        len(schema),
        len(schema) - len(filtered),
        len(filtered),
    )
    return filtered
