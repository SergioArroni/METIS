"""Configuration loading and validation."""

from pathlib import Path
from typing import Any

import yaml

from ...domain.errors import ConfigError


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

        return config

    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML syntax: {e}", config_path) from e
    except Exception as e:
        raise ConfigError(f"Failed to load configuration: {e}", config_path) from e


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
                    "'risk_aversion' must be positive (recommended: 5.0-7.0)",
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
