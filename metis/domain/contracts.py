"""Domain contracts — Protocol-based interfaces for the evaluation pipeline.

Each protocol defines a single responsibility in the pipeline:
  Load → Preprocess → Validate → Calibrate → Evaluate → Aggregate → Report
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

if TYPE_CHECKING:
    import pandas as pd

    from .entities import DatasetSpec, MetricResult, ReportSpec, RunSummary

# =============================================================================
# type Schema Contracts for Heterogeneous Data
# =============================================================================

SUPPORTED_TYPES = {
    "boolean",
    "categorical",
    "ordinal",
    "continuous",
    "discrete",
    "datetime",
    "geospatial",
    "text",
    "code_numeric",
    "id",
}


@dataclass
class ColumnTypeSpec:
    """Specification for a column's semantic type."""

    type: str
    levels: list[str] | None = None
    ranges: list[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported type '{self.type}'. Supported: {sorted(SUPPORTED_TYPES)}"
            )
        if self.type == "ordinal" and not self.levels:
            raise ValueError("Type 'ordinal' requires 'levels' list (order matters)")
        if self.type == "discrete" and not self.ranges:
            raise ValueError("type 'discrete' requires 'ranges' list of [low, high] pairs")
        if self.ranges:
            for r in self.ranges:
                if not isinstance(r, list | tuple) or len(r) != 2:
                    raise ValueError(f"Each range must be [low, high] pair, got: {r}")


@dataclass
class TypeSchema:
    """Schema mapping column names → semantic types."""

    columns: dict[str, str | ColumnTypeSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {}
        for col, spec in self.columns.items():
            if isinstance(spec, str):
                normalized[col] = ColumnTypeSpec(type=spec.strip().lower())
            elif isinstance(spec, dict):
                typ = spec.get("type", "").strip().lower()
                levels = spec.get("levels")
                ranges = spec.get("ranges")
                if ranges:
                    ranges = [(float(r[0]), float(r[1])) for r in ranges]
                normalized[col] = ColumnTypeSpec(type=typ, levels=levels, ranges=ranges)
            elif isinstance(spec, ColumnTypeSpec):
                normalized[col] = spec
            else:
                raise ValueError(f"Invalid type spec for column '{col}': {spec}")
        self.columns = normalized

    def get_type(self, column: str) -> str:
        spec = self.columns.get(column)
        if spec is None:
            raise KeyError(f"Column '{column}' not in schema")
        return spec.type

    def get_id_columns(self) -> list[str]:
        return [col for col, spec in self.columns.items() if spec.type == "id"]

    def get_non_id_columns(self) -> list[str]:
        return [col for col, spec in self.columns.items() if spec.type != "id"]


# =============================================================================
# Metric & Registry Contracts
# =============================================================================


class Metric(Protocol):
    """Strategy interface for a single metric computation."""

    name: str
    family: Literal["fidelity", "utility", "privacy"]
    purpose_tags: set[str]

    def fit(self, real: "pd.DataFrame", synth: "pd.DataFrame", context: dict) -> Self: ...

    def compute(self) -> "MetricResult": ...


class MetricRegistry(Protocol):
    """Registry for metric look-up."""

    def register(self, metric_id: str, metric_class: type[Metric]) -> None: ...
    def get(self, metric_id: str) -> type[Metric]: ...
    def list_ids(self, family: str | None = None) -> list[str]: ...


class Reporter(Protocol):
    """Renders a RunSummary into an output format (JSON, Markdown, …)."""

    def render(self, run_summary: "RunSummary", report_spec: "ReportSpec") -> None: ...


class ReporterRegistry(Protocol):
    def register(self, format_name: str, reporter_class: type[Reporter]) -> None: ...
    def get(self, format_name: str) -> type[Reporter]: ...
    def list_formats(self) -> list[str]: ...


class Preprocessor(Protocol):
    """Stateful data transformer (fit/transform)."""

    def fit(self, data: "pd.DataFrame", spec: "DatasetSpec") -> Self: ...
    def transform(self, data: "pd.DataFrame") -> "pd.DataFrame": ...
    def metadata(self) -> dict[str, Any]: ...


class PreprocessorRegistry(Protocol):
    def register(self, name: str, cls: type[Preprocessor]) -> None: ...
    def get(self, name: str) -> type[Preprocessor]: ...
    def list_names(self) -> list[str]: ...


# =============================================================================
# Calibration Contracts
# =============================================================================


class CalibrationStrategy(Protocol):
    """Upper/lower bound estimation strategy."""

    def calibrate(
        self,
        real_data: "pd.DataFrame",
        config_template_path: str,
        n_iterations: int,
        sample_size: int,
        base_seed: int,
        n_jobs: int,
    ) -> dict[str, list[float]]: ...

    def get_strategy_name(self) -> str: ...


class CalibrationEvaluator(Protocol):
    """Runs METIS evaluation within calibration loops.

    Returns a 3-tuple:
      - family_scores: ``{"fidelity": 0.85, ...}``
      - metric_values_by_family: per-metric normalized values grouped by family.
      - raw_values_by_family: per-metric pre-normalization values for traceability.
    """

    def evaluate(
        self,
        real_data: "pd.DataFrame",
        synth_data: "pd.DataFrame",
        seed: int,
    ) -> tuple[
        dict[str, float],
        dict[str, dict[str, float]],
        dict[str, dict[str, float]],
    ]: ...


class BoundsStorage(Protocol):
    """Calibration bounds persistence and normalisation."""

    def save(self, filepath: str) -> None: ...

    def set_bounds(
        self,
        family: str,
        lower_bound: float,
        upper_bound: float,
        lower_iterations: list[float],
        upper_iterations: list[float],
    ) -> None: ...

    def get_bounds(self, family: str) -> tuple[float, float]: ...
    def normalize_with_bounds(self, family: str, raw_value: float) -> float: ...


class NoiseGenerator(Protocol):
    """Generates synthetic noise data for lower-bound calibration."""

    def generate(
        self, reference_data: "pd.DataFrame", n_samples: int, seed: int
    ) -> "pd.DataFrame": ...
