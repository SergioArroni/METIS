"""
Result data structure for marginal metrics aggregation.
"""

from dataclasses import dataclass
from typing import Any

from .coverage import CoverageResult
from .scales import ScalesResult
from .tails import TailsResult


@dataclass
class MarginalResult:
    """
    Result of marginal fidelity aggregation.

    Combines results from three subcategories:
    - Tails: Distribution tail comparison
    - Scales: Central tendency and dispersion
    - Coverage: Categorical/coverage metrics

    Attributes:
        score: Final marginal score Q ∈ [0, 1]
        tails_score: Score from tails subcategory
        scales_score: Score from scales subcategory
        coverage_score: Score from coverage subcategory
        tails_result: Detailed tails result
        scales_result: Detailed scales result
        coverage_result: Detailed coverage result
        column_scores: Aggregated score per column
        subcategories_used: list of subcategories used
        n_columns: Number of columns processed
    """

    score: float

    # Subcategory scores
    tails_score: float
    scales_score: float
    coverage_score: float

    # Detailed results from each subcategory
    tails_result: TailsResult | None
    scales_result: ScalesResult | None
    coverage_result: CoverageResult | None

    # Column-level aggregated scores
    column_scores: dict[str, float]

    # Metadata
    subcategories_used: list[str]
    n_columns: int

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "score": self.score,
            "tails_score": self.tails_score,
            "scales_score": self.scales_score,
            "coverage_score": self.coverage_score,
            "column_scores": self.column_scores,
            "subcategories_used": self.subcategories_used,
            "n_columns": self.n_columns,
            "tails_result": self.tails_result.to_dict() if self.tails_result else None,
            "scales_result": (self.scales_result.to_dict() if self.scales_result else None),
            "coverage_result": (self.coverage_result.to_dict() if self.coverage_result else None),
        }

    def get_report_data(self) -> dict[str, Any]:
        """
        Get structured data for report generation.

        Returns:
            Dictionary with all reporting data
        """
        import numpy as np

        return {
            "metric_id": "fidelity.marginal",
            "score": self.score,
            "interpretation": self._interpret_score(),
            "subcategory_scores": {
                "tails": self.tails_score,
                "scales": self.scales_score,
                "coverage": self.coverage_score,
            },
            "column_scores": self.column_scores,
            "worst_columns": self._get_worst_columns(5),
            "subcategories_used": self.subcategories_used,
            "summary_stats": {
                "n_columns": self.n_columns,
                "mean_column_score": (
                    float(np.mean(list(self.column_scores.values()))) if self.column_scores else 0.0
                ),
            },
        }

    def _interpret_score(self) -> str:
        """Provide human-readable interpretation of the score."""
        if self.score >= 0.9:
            return "Excelente - Las distribuciones marginales son muy similares"
        if self.score >= 0.7:
            return "Bueno - Las distribuciones marginales son razonablemente similares"
        if self.score >= 0.5:
            return "Moderado - Algunas diferencias en las distribuciones marginales"
        if self.score >= 0.3:
            return "Pobre - Diferencias significativas en las distribuciones marginales"
        return "Muy pobre - Las distribuciones marginales son muy diferentes"

    def _get_worst_columns(self, n: int = 5) -> list[dict[str, Any]]:
        """Get the n columns with worst scores."""
        sorted_cols = sorted(self.column_scores.items(), key=lambda x: x[1])
        return [{"column": col, "score": score} for col, score in sorted_cols[:n]]

    @classmethod
    def empty(cls, subcategories_used: list[str]) -> "MarginalResult":
        """
        Create an empty result when no data is available.

        Args:
            subcategories_used: list of subcategory names

        Returns:
            Empty MarginalResult instance
        """
        return cls(
            score=0.0,
            tails_score=0.0,
            scales_score=0.0,
            coverage_score=0.0,
            tails_result=None,
            scales_result=None,
            coverage_result=None,
            column_scores={},
            subcategories_used=subcategories_used,
            n_columns=0,
        )
