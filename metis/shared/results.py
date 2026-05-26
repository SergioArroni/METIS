"""
Result data structures for marginal metrics.

Provides common result classes used across all marginal metric types.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ColumnMetricResult:
    """
    Result of a metric computation for a single column.

    Stores both the raw metric value and a normalized version in [0, 1]
    where 1 indicates best quality. The normalization handles the inversion
    for distance metrics (where lower raw values are better).

    Attributes:
        column: Name of the column this result refers to
        raw_value: Original metric value before normalization
        normalized_value: Value in [0, 1] where 1 = best quality
        is_valid: Whether the computation was successful
        error: Error message if computation failed

    Example:
        >>> result = ColumnMetricResult(
        ...     column="age", raw_value=0.05, normalized_value=0.95, is_valid=True
        ... )
        >>> result.is_valid
        True
    """

    column: str
    raw_value: float
    normalized_value: float  # In [0, 1], where 1 = best
    is_valid: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "column": self.column,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "is_valid": self.is_valid,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnMetricResult":
        """
        Create from dictionary.

        Args:
            data: Dictionary with result data

        Returns:
            ColumnMetricResult instance
        """
        return cls(
            column=data["column"],
            raw_value=data["raw_value"],
            normalized_value=data["normalized_value"],
            is_valid=data["is_valid"],
            error=data.get("error"),
        )

    @classmethod
    def invalid(cls, column: str, error: str) -> "ColumnMetricResult":
        """
        Create an invalid result with error message.

        Args:
            column: Column name
            error: Error description

        Returns:
            Invalid ColumnMetricResult instance
        """
        return cls(
            column=column,
            raw_value=float("nan"),
            normalized_value=0.0,
            is_valid=False,
            error=error,
        )
