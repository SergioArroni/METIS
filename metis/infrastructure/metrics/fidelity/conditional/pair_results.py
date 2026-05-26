"""
Result data structures for conditional metrics.

Provides common result classes used across all conditional metric types.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PairMetricResult:
    """
    Result of a metric computation for a column pair.

    Stores both the real and synthetic metric values, their difference,
    and a normalized version in [0, 1] where 1 indicates best quality.

    Attributes:
        col1: Name of the first column
        col2: Name of the second column
        real_value: Metric value computed on real data
        synth_value: Metric value computed on synthetic data
        delta: Absolute difference between real and synthetic values
        normalized_value: Value in [0, 1] where 1 = best quality
        is_valid: Whether the computation was successful
        error: Error message if computation failed

    Example:
        >>> result = PairMetricResult(
        ...     col1="age",
        ...     col2="income",
        ...     real_value=0.85,
        ...     synth_value=0.82,
        ...     delta=0.03,
        ...     normalized_value=0.97,
        ...     is_valid=True,
        ... )
        >>> result.is_valid
        True
    """

    col1: str
    col2: str
    real_value: float
    synth_value: float
    delta: float  # Absolute difference
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
            "col1": self.col1,
            "col2": self.col2,
            "real_value": self.real_value,
            "synth_value": self.synth_value,
            "delta": self.delta,
            "normalized_value": self.normalized_value,
            "is_valid": self.is_valid,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PairMetricResult":
        """
        Create from dictionary.

        Args:
            data: Dictionary with result data

        Returns:
            PairMetricResult instance
        """
        return cls(
            col1=data["col1"],
            col2=data["col2"],
            real_value=data["real_value"],
            synth_value=data["synth_value"],
            delta=data["delta"],
            normalized_value=data["normalized_value"],
            is_valid=data["is_valid"],
            error=data.get("error"),
        )

    @classmethod
    def invalid(cls, col1: str, col2: str, error: str) -> "PairMetricResult":
        """
        Create an invalid result with error message.

        Args:
            col1: First column name
            col2: Second column name
            error: Error description

        Returns:
            Invalid PairMetricResult instance
        """
        return cls(
            col1=col1,
            col2=col2,
            real_value=float("nan"),
            synth_value=float("nan"),
            delta=float("nan"),
            normalized_value=0.0,
            is_valid=False,
            error=error,
        )
