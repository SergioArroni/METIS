"""
Shared utilities for GAN-based generators (CTGAN, ADSGAN).
"""

import warnings

import pandas as pd

DEFAULT_HIGH_CARDINALITY_THRESHOLD = 100


def filter_high_cardinality(
    data: pd.DataFrame,
    categorical_columns: list[str],
    threshold: int = DEFAULT_HIGH_CARDINALITY_THRESHOLD,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Remove categorical columns whose cardinality exceeds *threshold* to
    prevent OOM when SDV one-hot encodes them.

    Returns:
        (filtered_data, kept_cat_cols, dropped_cols)
    """
    dropped: list[str] = []
    kept: list[str] = []
    for col in categorical_columns:
        if col in data.columns and data[col].nunique() > threshold:
            dropped.append(col)
        else:
            kept.append(col)

    if dropped:
        est_cols = sum(data[c].nunique() for c in dropped)
        warnings.warn(
            f"CTGAN/ADSGAN: dropping {len(dropped)} high-cardinality columns "
            f"(>{threshold} unique values, ~{est_cols} one-hot columns avoided): "
            f"{dropped}.  These columns will be filled from the real-data "
            f"distribution after generation.",
            stacklevel=2,
        )
        data = data.drop(columns=dropped)

    return data, kept, dropped


# Label for the categories outside the kept top-k of a high-cardinality column.
OTHER_CATEGORY = "OTHER"


def collapse_high_cardinality(
    data: pd.DataFrame,
    categorical_columns: list[str],
    threshold: int = DEFAULT_HIGH_CARDINALITY_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep the *threshold* most frequent categories of each high-cardinality
    column and map the rest to ``OTHER_CATEGORY``, so the model still learns the
    column jointly with the others without one-hot exploding.

    The previous approach dropped those columns and refilled them afterwards by
    resampling the *real* values: real identifiers (Airbnb ``name``,
    ``host_name``, ``license``) reached the synthetic table verbatim and the
    column lost every dependence on the rest.  METIS reads such text columns as
    top-k + ``OTHER`` too.

    Returns:
        (collapsed_data, collapsed_cols)
    """
    data = data.copy()
    collapsed: list[str] = []
    for col in categorical_columns:
        if col in data.columns and data[col].nunique() > threshold:
            top = data[col].value_counts().index[:threshold]
            data[col] = data[col].where(data[col].isin(top) | data[col].isna(), OTHER_CATEGORY)
            collapsed.append(col)
    return data, collapsed
