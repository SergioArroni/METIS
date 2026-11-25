"""Data and configuration fingerprinting for calibration cache validation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from metis.shared.normalization import METRIC_NORMALIZATION_MAP

# 3: auditoria 2026-10-08 (escala absoluta de fidelidad, agregacion jerarquica,
# privacidad relativa): las cotas calibradas antes no son comparables.
# 4: utilidad sin Optuna y en escala absoluta (2026-10-09).
# 5: cotas por métrica (fidelidad: real no vista / ruido; privacidad: real no
#    vista / copia exacta, con holdout; utilidad sin calibrar) (2026-10-09).
# 6: t_closeness con la EMD de Li et al. (distancia ordenada para columnas
#    sensibles numéricas y ordinales) (2026-10-14).
# 7: privacidad revisada: control = holdout frente a la misma sintética,
#    mia/record_linkage/inference_attack auditados en la cola (ε), dcr y nnaa
#    con el holdout como control (2026-10-14).
# 8: la calibración decide una vez por tabla si la utilidad aplica
#    (20 medias muestras; ``utility_signal``) (2026-10-18).
# 9: la medida por tabla fija también las anclas de la utilidad (2026-10-18).
# 10: privacidad con el vocabulario categórico de todas las tablas comparadas
#     (las columnas casi únicas acercaban la sintética al holdout) (2026-10-18).
# 11: Theil's U en las dos direcciones de cada par (2026-10-19).
# 12: utilidad de clasificación con la ROC AUC macro uno-contra-resto (2026-10-19).
# 13: Cramér's V, ω² y dCor con corrección de sesgo, energy como coeficiente H,
#     utilidad exige holdout, δ de privacidad diferencial como condición (2026-10-19).
# 14: reparto 60/20/20, real y sintética al mismo tamaño, utilidad simétrica (2026-10-19).
# 15: privacidad con la subcategoría «attacks» (mia, inference_attack, record_linkage) (2026-10-19).
# 16: los cuatro tipos de la Fase −1 (discreta con sombra __lvl) (2026-10-19).
# 17: la discreta sin sombra: su dominio es una regla del eje de restricciones (2026-10-20).
# 18: la utilidad aplica si la ganancia de TTR supera 3 (antes 5) veces su ruido (2026-10-25).
# 19: fidelity.discriminator (categoría detection) y utility.log_loss con su
#     propia señal y anclas (2026-11-03).
CALIBRATION_CACHE_SCHEMA_VERSION = 21  # 21: attacks also try partial exact matches; 20: calibration split = benchmark split


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
        — each component is a 32-char (128-bit) prefix of SHA-256, enough
        to resist collision attacks while keeping cache filenames inside
        OS path-length limits (notably Windows MAX_PATH).
    """
    del sample_size  # accepted for backwards-compat, not used

    # Component 1: Column structure hash
    columns_str = "|".join(sorted(df.columns))
    columns_hash = hashlib.sha256(columns_str.encode()).hexdigest()[:32]

    # Component 2: Full content hash. ``hash_pandas_object`` is row-wise
    # deterministic and tolerant of mixed dtypes; fall back to CSV bytes
    # only if it cannot handle a column type.
    try:
        content_bytes = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    except (TypeError, ValueError):
        content_bytes = df.to_csv(index=False).encode()

    content_hash = hashlib.sha256(content_bytes).hexdigest()[:32]

    # Component 3: Shape hash
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    shape_hash = hashlib.sha256(shape_str.encode()).hexdigest()[:32]

    return f"{columns_hash}_{content_hash}_{shape_hash}"


def compute_config_fingerprint(config_path: str) -> str:
    """
    Generate hash of metric configuration.

    Only hashes the parts of config that affect calibration:
    - evaluation.metric_ids: Which metrics are computed
    - data.task_type: Classification vs regression
    - data.schema: Column definitions
    - normalization mapping: How raw metric values are oriented/scaled
    - when set: data.target, evaluation.metric_params,
      evaluation.ml_efficiency_config, aggregation.weights and
      aggregation.risk_aversion (they change metric values or family scores)

    Args:
        config_path: Path to YAML configuration file

    Returns:
        SHA256 hash (first 32 chars) of relevant config sections

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
    # Settings that change metric values or family scores, hence the bounds.
    # Added only when set, so the key of a config without them is unchanged.
    data_cfg = config.get("data") or {}
    eval_cfg = config.get("evaluation") or {}
    agg_cfg = config.get("aggregation") or {}
    optional = {
        "target": data_cfg.get("target"),
        "metric_params": {k: v for k, v in (eval_cfg.get("metric_params") or {}).items() if v},
        "ml_efficiency_config": eval_cfg.get("ml_efficiency_config"),
        "aggregation_weights": agg_cfg.get("weights"),
        "risk_aversion": agg_cfg.get("risk_aversion"),
        # The split the calibration uses: its own or, without it, the benchmark's
        # (``metis.shared.splits.split_fractions``).
        "calibration_split": (config.get("calibration") or {}).get("split")
        or (config.get("benchmark") or {}).get("split"),
    }
    relevant_config.update({k: v for k, v in optional.items() if v not in (None, {}, [])})

    # Create deterministic JSON representation
    config_str = json.dumps(relevant_config, sort_keys=True, indent=None)

    return hashlib.sha256(config_str.encode()).hexdigest()[:32]


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
    params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]

    return f"calibration_{data_fingerprint}_{config_fingerprint}_{params_hash}"
