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

# The closed set of variable types of Hefesto's Phase −1 (rules I-IV: nominal,
# ordinal, continuous numeric, discrete numeric), plus ``id`` for the columns
# excluded from evaluation (question Q0: identifiers, non-tabular blobs).
# ``metis.infrastructure.typing.infer_schema`` infers them.
SUPPORTED_TYPES = {"nominal", "ordinal", "continuous", "discrete", "id"}

# Types of METIS 1.x and the Phase −1 type each one became (2026-10-19).
_RENAMED_TYPES = {
    "categorical": "nominal",
    "boolean": "nominal",
    "text": "nominal",
    "code_numeric": "nominal",
    "datetime": "continuous (with datetime: true)",
    "geospatial": "continuous",
}


@dataclass
class ColumnTypeSpec:
    """Specification for a column's semantic type.

    - ``nominal``: categories without order.
    - ``ordinal``: categories with an order, ``levels`` from first to last.
    - ``continuous``: any value within the range; ``datetime: true`` for
      dates and timestamps (measured in seconds).
    - ``discrete``: numeric on a closed set (``levels``, the enumerated
      domain) or on a lattice of step ``step``; with neither, the domain is
      the real column's values when there are few of them.
    - ``id``: excluded from evaluation.
    """

    type: str
    levels: list | None = None
    step: float | None = None
    datetime: bool = False

    def __post_init__(self) -> None:
        if self.type in _RENAMED_TYPES:
            raise ValueError(
                f"Type '{self.type}' no longer exists: METIS uses the four types of "
                f"Phase −1 {sorted(SUPPORTED_TYPES - {'id'})} (and 'id'); use "
                f"'{_RENAMED_TYPES[self.type]}'"
            )
        if self.type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported type '{self.type}'. Supported: {sorted(SUPPORTED_TYPES)}"
            )
        if self.type == "ordinal" and not self.levels:
            raise ValueError("Type 'ordinal' requires 'levels' list (order matters)")
        if self.step is not None and (self.type != "discrete" or not float(self.step) > 0):
            raise ValueError("'step' is a positive lattice step of a 'discrete' column")
        if self.datetime and self.type != "continuous":
            raise ValueError("'datetime: true' only applies to 'continuous' columns")


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
                if "ranges" in spec:
                    raise ValueError(
                        f"Column '{col}': 'ranges' no longer exists; a 'discrete' column "
                        "takes 'levels' (its values) or 'step' (its lattice)"
                    )
                normalized[col] = ColumnTypeSpec(
                    type=spec.get("type", "").strip().lower(),
                    levels=spec.get("levels"),
                    step=float(spec["step"]) if spec.get("step") is not None else None,
                    datetime=bool(spec.get("datetime", False)),
                )
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

    def fit(self, real: pd.DataFrame, synth: pd.DataFrame, context: dict) -> Self: ...

    def compute(self) -> MetricResult: ...


class MetricRegistry(Protocol):
    """Registry for metric look-up."""

    def register(self, metric_id: str, metric_class: type[Metric]) -> None: ...
    def get(self, metric_id: str) -> type[Metric]: ...
    def list_ids(self, family: str | None = None) -> list[str]: ...


class Reporter(Protocol):
    """Renders a RunSummary into an output format (JSON, Markdown, …)."""

    def render(self, run_summary: RunSummary, report_spec: ReportSpec) -> None: ...


class ReporterRegistry(Protocol):
    def register(self, format_name: str, reporter_class: type[Reporter]) -> None: ...
    def get(self, format_name: str) -> type[Reporter]: ...
    def list_formats(self) -> list[str]: ...


class Preprocessor(Protocol):
    """Stateful data transformer (fit/transform)."""

    def fit(self, data: pd.DataFrame, spec: DatasetSpec) -> Self: ...
    def transform(self, data: pd.DataFrame) -> pd.DataFrame: ...
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
        real_data: pd.DataFrame,
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
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        seed: int,
    ) -> tuple[
        dict[str, float],
        dict[str, dict[str, float]],
        dict[str, dict[str, float]],
    ]: ...


class BoundsStorage(Protocol):
    """Per-metric calibration bounds: persistence and normalisation."""

    def save(self, filepath: str) -> None: ...

    def set_bounds(
        self,
        metric_id: str,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        lower_iterations: list[float] | None = None,
        upper_iterations: list[float] | None = None,
    ) -> None: ...

    def __contains__(self, metric_id: str) -> bool: ...
    def get_bounds(self, metric_id: str) -> tuple[float, float]: ...
    def is_calibratable(self, metric_id: str) -> bool: ...
    def exclusion_reason(self, metric_id: str) -> str | None: ...
    def normalize_with_bounds(self, metric_id: str, raw_value: float) -> float: ...


class NoiseGenerator(Protocol):
    """Generates synthetic noise data for lower-bound calibration."""

    def generate(
        self, reference_data: pd.DataFrame, n_samples: int, seed: int
    ) -> pd.DataFrame: ...
