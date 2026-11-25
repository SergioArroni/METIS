"""Configuration loading and validation."""

import logging
import warnings
from pathlib import Path
from typing import Any

import yaml

from ...domain.errors import ConfigError

_logger = logging.getLogger(__name__)

# Keys METIS reads.  ``None`` means "free-form below this level".
KNOWN_CONFIG_KEYS: dict[str, set[str] | None] = {
    "data": {
        "real",
        "synthetic",
        "holdout",
        "target",
        "task_type",
        "schema",
        "dtypes",
        "constraints",
        "separator",
        "real_separator",
        "synth_separator",
        "holdout_separator",
    },
    "metrics": None,
    "evaluation": {
        "n_runs",
        "seed",
        "metric_ids",
        "cv_splits",
        "ml_efficiency_config",
        "metric_params",
    },
    "reproducibility": {"seed"},
    "calibration": {
        "mode",
        "enabled",
        "bounds_file",
        "aggregators_file",
        "force",
        "base_seed",
        "n_iterations",
        "sample_percentage",
        "n_jobs",
        "tune_aggregators",
        "cache_dir",
    },
    "aggregation": {"risk_aversion", "weights"},
    "report": {"formats", "output_dir", "include_details", "include_artifacts"},
    "benchmark": None,
}


class UnknownConfigKeyWarning(UserWarning):
    """A config key METIS does not read (probably a typo)."""


def find_unknown_config_keys(config: dict[str, Any]) -> list[str]:
    """Dotted paths of keys in *config* that METIS never reads."""
    unknown: list[str] = []
    for key, value in config.items():
        if key not in KNOWN_CONFIG_KEYS:
            unknown.append(str(key))
            continue
        allowed = KNOWN_CONFIG_KEYS[key]
        if allowed is None or not isinstance(value, dict):
            continue
        unknown.extend(f"{key}.{sub}" for sub in value if sub not in allowed)
    return unknown


def check_config_keys(
    config: dict[str, Any], *, strict: bool = False, source: str = "config"
) -> list[str]:
    """Warn (or raise with ``strict=True``) about keys METIS ignores.

    Returns the list of human-readable warnings, also emitted as
    :class:`UnknownConfigKeyWarning` and logged.
    """
    unknown = find_unknown_config_keys(config)
    if not unknown:
        return []
    msg = (
        f"Unknown key(s) in {source}, ignored by METIS: {unknown}. "
        f"Known sections: {sorted(KNOWN_CONFIG_KEYS)}"
    )
    if strict:
        raise ConfigError(msg)
    _logger.warning(msg)
    warnings.warn(msg, UnknownConfigKeyWarning, stacklevel=3)
    return [msg]


def validate_dataframe_config(config: Any, *, strict: bool = False) -> list[str]:
    """Structural checks for a config passed to the SDK (no file paths needed).

    Requires a ``data`` mapping with a ``schema`` mapping and a metric list
    (``metrics`` or ``evaluation.metric_ids``); checks the type of the
    optional sections and of ``evaluation.metric_params``; and reports
    unknown keys (see :func:`check_config_keys`).
    """
    if not isinstance(config, dict):
        raise ConfigError(f"config must be a dict, got {type(config).__name__}")
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ConfigError("'data' section must be a dictionary")
    schema = data_cfg.get("schema")
    if schema != "auto" and (not isinstance(schema, dict) or not schema):
        raise ConfigError(
            "'data.schema' must be a non-empty mapping {column: type}, or 'auto' to infer "
            "the types of the real table with the Phase −1 rules"
        )
    for section in ("evaluation", "reproducibility", "calibration", "aggregation", "report"):
        if section in config and not isinstance(config[section], dict):
            raise ConfigError(f"'{section}' section must be a dictionary")
    eval_cfg = config.get("evaluation") or {}
    metrics = config.get("metrics") or eval_cfg.get("metric_ids")
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("'metrics' must be a non-empty list")
    _check_metric_params(eval_cfg.get("metric_params"))
    return check_config_keys(config, strict=strict, source="config")


def _check_metric_params(metric_params: Any, config_path: str | None = None) -> None:
    if metric_params is None:
        return
    if not isinstance(metric_params, dict):
        raise ConfigError(
            "'evaluation.metric_params' must map metric ids to parameter mappings", config_path
        )
    for metric_id, params in metric_params.items():
        if not isinstance(params, dict):
            raise ConfigError(
                f"'evaluation.metric_params.{metric_id}' must be a mapping, got {params!r}",
                config_path,
            )


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load and validate YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        ConfigError: If config file cannot be loaded or is invalid
    """
    try:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}", config_path)

        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ConfigError("Configuration file is empty", config_path)

        # Basic validation
        _validate_config_structure(config, config_path)

    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML syntax: {e}", config_path) from e
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"Failed to load configuration: {e}", config_path) from e

    check_config_keys(config, source=str(config_path))
    return config


def _validate_config_structure(config: dict[str, Any], config_path: str) -> None:
    """Validate basic configuration structure."""

    # Required top-level sections
    required_sections = ["data"]
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required section: {section}", config_path)

    # Validate data section
    data_config = config["data"]
    if not isinstance(data_config, dict):
        raise ConfigError("'data' section must be a dictionary", config_path)

    required_data_keys = ["real", "synthetic"]
    for key in required_data_keys:
        if key not in data_config:
            raise ConfigError(f"Missing required key in data section: {key}", config_path)

    # Validate metrics: top-level 'metrics' or legacy 'evaluation.metric_ids'
    has_top_metrics = "metrics" in config and isinstance(config["metrics"], list)
    has_eval_metrics = isinstance(config.get("evaluation"), dict) and "metric_ids" in config.get(
        "evaluation", {}
    )
    if not has_top_metrics and not has_eval_metrics:
        raise ConfigError(
            "'metrics' list must be specified in config",
            config_path,
        )

    # Validate evaluation section (if present)
    if "evaluation" in config:
        eval_config = config["evaluation"]
        if not isinstance(eval_config, dict):
            raise ConfigError("'evaluation' section must be a dictionary", config_path)
        _check_metric_params(eval_config.get("metric_params"), config_path)

    # Validate report section (if present)
    if "report" in config:
        report_config = config["report"]
        if not isinstance(report_config, dict):
            raise ConfigError("'report' section must be a dictionary", config_path)

        if "formats" not in report_config:
            raise ConfigError("'formats' must be specified in report section", config_path)

        if "output_dir" not in report_config:
            raise ConfigError("'output_dir' must be specified in report section", config_path)

    # Validate aggregation section (if present)
    if "aggregation" in config:
        agg_config = config["aggregation"]
        if not isinstance(agg_config, dict):
            raise ConfigError("'aggregation' section must be a dictionary", config_path)

        # Validate risk_aversion parameter if present
        if "risk_aversion" in agg_config:
            risk_aversion = agg_config["risk_aversion"]
            if not isinstance(risk_aversion, int | float):
                raise ConfigError("'risk_aversion' must be a number", config_path)
            if risk_aversion <= 0:
                raise ConfigError(
                    "'risk_aversion' must be positive (METIS default: 3)",
                    config_path,
                )


def save_config(config: dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        config_path: Output file path
    """
    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=True)

    except Exception as e:
        raise ConfigError(f"Failed to save configuration: {e}", config_path) from e


def merge_configs(base_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two configuration dictionaries with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged
