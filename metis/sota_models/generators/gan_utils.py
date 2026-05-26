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
