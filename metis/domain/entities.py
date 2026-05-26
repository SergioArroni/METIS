"""Domain entities using dataclasses for immutable data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd

    from .contracts import TypeSchema


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for dataset structure and constraints."""

    target: str | list[str] | None = None
    task_type: Literal["classification", "regression"] | None = None
    dtypes: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dataset specification after initialization."""
        if self.target is not None:
            if isinstance(self.target, str):
                pass  # single target
            elif isinstance(self.target, list):
                if not self.target:
                    raise ValueError("Target list cannot be empty")
                if not all(isinstance(t, str) for t in self.target):
                    raise ValueError("All targets must be string column names")
            else:
                raise ValueError("Target must be a string, list of strings, or None")
        if self.task_type and self.task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'")

    @property
    def target_list(self) -> list[str]:
        """Return target(s) as a list, normalising single-string and None."""
        if self.target is None:
            return []
        if isinstance(self.target, str):
            return [self.target]
        return list(self.target)


@dataclass(frozen=True)
class EvalPlan:
    """Evaluation plan specifying metrics and execution parameters."""

    metric_ids: list[str]
    seed: int = 42
    cv_splits: int = 3

    def __post_init__(self) -> None:
        """Validate evaluation plan after initialization."""
        if not self.metric_ids:
            raise ValueError("At least one metric_id must be specified")
        if self.seed < 0:
            raise ValueError("Seed must be non-negative")
        if self.cv_splits < 2:
            raise ValueError("CV splits must be at least 2")


@dataclass(frozen=True)
class MetricResult:
    """Result from a single metric computation."""

    id: str
    value: float
    details: dict[str, Any]
    family: Literal["fidelity", "utility", "privacy"]
    purpose_tags: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate metric result after initialization."""
        if not self.id:
            raise ValueError("Metric ID cannot be empty")
        if self.family not in {"fidelity", "utility", "privacy"}:
            raise ValueError(f"Invalid family: {self.family}")


@dataclass(frozen=True)
class RunSummary:
    """Summary of a complete evaluation run."""

    plan: EvalPlan
    results: list[MetricResult]
    aggregates: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)

    def get_results_by_family(self, family: str) -> list[MetricResult]:
        """Get all results for a specific family."""
        return [r for r in self.results if r.family == family]

    def get_family_score(self, family: str) -> float:
        """Get aggregated score for a specific family."""
        return self.aggregates.get(f"{family}_score", 0.0)


@dataclass(frozen=True)
class ReportSpec:
    """Specification for report generation."""

    formats: list[str]
    output_dir: str
    include_details: bool = True
    include_artifacts: bool = False

    def __post_init__(self) -> None:
        """Validate report specification after initialization."""
        if not self.formats:
            raise ValueError("At least one format must be specified")
        if not self.output_dir:
            raise ValueError("Output directory cannot be empty")


@dataclass
class TransformedData:
    """
    Container for transformed heterogeneous data.

    After applying SimpleCaster, data is split into:
    - cat: DataFrame with categorical columns (strings)
    - num: DataFrame with numeric columns (floats)
    - full: Concatenation of cat + num for metrics that need all data

    Attributes:
        cat: DataFrame with categorical columns
        num: DataFrame with numeric columns
        full: Concatenated DataFrame (cat + num)
        meta: Metadata from transformation (column types, mappings, etc.)
        excluded_ids: list of column names that were declared as 'id' and excluded
        schema: The TypeSchema that was applied
    """

    cat: pd.DataFrame
    num: pd.DataFrame
    full: pd.DataFrame
    meta: dict[str, Any]
    excluded_ids: list[str]
    schema: TypeSchema | None = None

    def get_column_types(self) -> dict[str, str]:
        """Get mapping of column name to semantic type."""
        result = {}
        for col, info in self.meta.items():
            if col.startswith("_"):  # Skip internal keys
                continue
            if isinstance(info, dict) and "type" in info:
                result[col] = info["type"]
        return result
