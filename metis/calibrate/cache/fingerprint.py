"""Data and configuration fingerprinting for calibration cache validation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from metis.shared.normalization import METRIC_NORMALIZATION_MAP

CALIBRATION_CACHE_SCHEMA_VERSION = 2


def compute_data_fingerprint(df: pd.DataFrame, sample_size: int | None = None) -> str:
    """
    Generate deterministic hash of DataFrame content.

    Uses a three-component hash:
    1. Column names hash (structure validation)
    2. Full content hash via :func:`pandas.util.hash_pandas_object` over
       every row — this defends against cache poisoning by datasets that
       differ only in non-sampled rows.
    3. Shape hash (dimensions validation)

    Args:
        df: DataFrame to fingerprint
        sample_size: Deprecated. Kept for backwards-compat with old call
            sites that passed it positionally; ignored. The fingerprint is
            now computed over the full DataFrame.

    Returns:
        Composite fingerprint string: "{columns_hash}_{content_hash}_{shape_hash}"
        — each component is a 16-char (64-bit) prefix of SHA-256, enough
        to resist collision attacks while keeping cache filenames inside
        OS path-length limits (notably Windows MAX_PATH ≈ 260).
    """
    del sample_size  # accepted for backwards-compat, not used

    # Component 1: Column structure hash
    columns_str = "|".join(sorted(df.columns))
    columns_hash = hashlib.sha256(columns_str.encode()).hexdigest()[:16]

    # Component 2: Full content hash. ``hash_pandas_object`` is row-wise
    # deterministic and tolerant of mixed dtypes; fall back to CSV bytes
    # only if it cannot handle a column type.
    try:
        content_bytes = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    except (TypeError, ValueError):
        content_bytes = df.to_csv(index=False).encode()

    content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]

    # Component 3: Shape hash
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    shape_hash = hashlib.sha256(shape_str.encode()).hexdigest()[:16]

    return f"{columns_hash}_{content_hash}_{shape_hash}"


def compute_config_fingerprint(config_path: str) -> str:
    """
    Generate hash of metric configuration.

    Only hashes the parts of config that affect calibration:
    - evaluation.metric_ids: Which metrics are computed
    - data.task_type: Classification vs regression
    - data.schema: Column definitions
    - normalization mapping: How raw metric values are oriented/scaled

    Args:
        config_path: Path to YAML configuration file

    Returns:
        SHA256 hash (first 16 chars) of relevant config sections

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config is malformed
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_file.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    normalization_signature = {
        metric_id: getattr(normalization_type, "value", str(normalization_type))
        for metric_id, normalization_type in sorted(METRIC_NORMALIZATION_MAP.items())
    }

    # Extract only calibration-relevant parts
    # Support new top-level 'metrics' list and legacy 'evaluation.metric_ids'
    metric_ids = config.get("metrics") or config.get("evaluation", {}).get("metric_ids", [])
    relevant_config: dict[str, Any] = {
        "cache_schema_version": CALIBRATION_CACHE_SCHEMA_VERSION,
        "metric_ids": sorted(metric_ids),
        "normalization_signature": normalization_signature,
        "task_type": config.get("data", {}).get("task_type"),
        "schema": config.get("data", {}).get("schema"),
    }

    # Create deterministic JSON representation
    config_str = json.dumps(relevant_config, sort_keys=True, indent=None)

    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def generate_cache_key(
    data_fingerprint: str,
    config_fingerprint: str,
    n_iterations: int,
    sample_size: int,
    base_seed: int,
) -> str:
    """
    Generate unique cache key for calibration results.

    Cache key format: calibration_{data_fp}_{config_fp}_{params_hash}

    Args:
        data_fingerprint: Fingerprint of input data
        config_fingerprint: Fingerprint of config
        n_iterations: Number of calibration iterations
        sample_size: Sample size per iteration
        base_seed: Base random seed

    Returns:
        Unique cache key string

    Examples:
        >>> key = generate_cache_key("abc123def456", "xyz789", 12, 750, 42)
        >>> key.startswith("calibration_")
        True
    """
    params_str = f"{n_iterations}_{sample_size}_{base_seed}"
    params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]

    return f"calibration_{data_fingerprint}_{config_fingerprint}_{params_hash}"
