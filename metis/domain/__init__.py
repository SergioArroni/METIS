"""Domain layer: Core contracts, entities, and business logic."""

from .contracts import (
    BoundsStorage,
    CalibrationEvaluator,
    CalibrationStrategy,
    ColumnTypeSpec,
    Metric,
    MetricRegistry,
    NoiseGenerator,
    Preprocessor,
    PreprocessorRegistry,
    Reporter,
    ReporterRegistry,
    TypeSchema,
)
from .entities import DatasetSpec, EvalPlan, MetricResult, ReportSpec, RunSummary, TransformedData
from .errors import ConfigError, SchemaError
from .taxonomy import FAMILIES

__all__ = [
    # Contracts
    "Metric",
    "Preprocessor",
    "Reporter",
    "MetricRegistry",
    "ReporterRegistry",
    "PreprocessorRegistry",
    "BoundsStorage",
    "TypeSchema",
    "ColumnTypeSpec",
    # Calibration contracts
    "CalibrationStrategy",
    "CalibrationEvaluator",
    "NoiseGenerator",
    # Entities
    "DatasetSpec",
    "EvalPlan",
    "MetricResult",
    "RunSummary",
    "ReportSpec",
    "TransformedData",
    # Taxonomy
    "FAMILIES",
    # Errors
    "ConfigError",
    "SchemaError",
]
