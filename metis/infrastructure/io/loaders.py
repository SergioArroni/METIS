"""Data loading utilities for various file formats."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...domain.errors import SchemaError


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame with basic preprocessing.

    Args:
        file_path: Path to CSV file
        **kwargs: Additional parameters passed to pd.read_csv

    Returns:
        Loaded DataFrame

    Raises:
        SchemaError: If file cannot be loaded or is invalid
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise SchemaError(f"File not found: {file_path}")

        # Default CSV loading parameters
        default_params = {
            "na_values": ["", "NA", "N/A", "null", "NULL", "None"],
            "keep_default_na": True,
            "skipinitialspace": True,
        }

        # Merge with user parameters (user params take precedence)
        params = {**default_params, **kwargs}

        # Load the CSV
        df = pd.read_csv(file_path, **params)

        # Basic validation
        if df.empty:
            raise SchemaError(f"Loaded DataFrame is empty: {file_path}")

        if len(df.columns) == 0:
            raise SchemaError(f"DataFrame has no columns: {file_path}")

        return _basic_cleanup(df)

    except pd.errors.EmptyDataError as e:
        raise SchemaError(f"CSV file is empty: {file_path}") from e
    except pd.errors.ParserError as e:
        raise SchemaError(f"Failed to parse CSV file {file_path}: {e}") from e
    except Exception as e:
        raise SchemaError(f"Failed to load CSV file {file_path}: {e}") from e


def load_parquet(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load Parquet file into pandas DataFrame.

    Args:
        file_path: Path to Parquet file
        **kwargs: Additional parameters passed to pd.read_parquet

    Returns:
        Loaded DataFrame

    Raises:
        SchemaError: If file cannot be loaded
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise SchemaError(f"File not found: {file_path}")

        df = pd.read_parquet(file_path, **kwargs)

        if df.empty:
            raise SchemaError(f"Loaded DataFrame is empty: {file_path}")

        return df

    except Exception as e:
        raise SchemaError(f"Failed to load Parquet file {file_path}: {e}") from e


def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        file_path: Output file path
        **kwargs: Additional parameters passed to df.to_csv
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        default_params = {"index": False, "na_rep": "NA"}

        params = {**default_params, **kwargs}
        df.to_csv(file_path, **params)

    except Exception as e:
        raise SchemaError(f"Failed to save CSV file {file_path}: {e}") from e


def _basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic DataFrame cleanup.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Strip whitespace from string columns
    string_columns = df.select_dtypes(include=["object"]).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()
        # Replace empty strings with NaN
        df[col] = df[col].replace("", np.nan)

    return df


def get_file_info(file_path: str) -> dict[str, Any]:
    """
    Get basic information about a data file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return {"exists": False, "error": "File not found"}

        info = {"exists": True, "size_bytes": path.stat().st_size, "suffix": path.suffix.lower()}

        # Try to get more details for known formats
        if info["suffix"] == ".csv":
            try:
                # Quick peek at the file
                sample = pd.read_csv(file_path, nrows=5)
                info.update(
                    {
                        "columns": list(sample.columns),
                        "n_columns": len(sample.columns),
                        "dtypes": {col: str(dtype) for col, dtype in sample.dtypes.items()},
                    }
                )
            except Exception as e:
                info["error"] = f"Could not read CSV sample: {e}"

        return info

    except Exception as e:
        return {"exists": False, "error": str(e)}
