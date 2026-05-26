"""
Result data structure for conditional metrics aggregation.
"""

from dataclasses import dataclass
from typing import Any

from .pair_results import PairMetricResult


@dataclass
class ConditionalResult:
    """
    Result of conditional fidelity aggregation.

    Contains aggregated scores for each category of conditional metrics
    and detailed results for each column pair.

    Attributes:
        score: Overall aggregated score in [0, 1]
        num_num_score: Aggregated score for Num↔Num metrics
        num_cat_score: Aggregated score for Num↔Cat metrics
        cat_cat_score: Aggregated score for Cat↔Cat metrics
        num_num_details: Detailed results for Num↔Num metrics
        num_cat_details: Detailed results for Num↔Cat metrics
        cat_cat_details: Detailed results for Cat↔Cat metrics
        n_pairs_computed: Total number of pairs computed
    """

    score: float

    num_num_score: float
    num_cat_score: float
    cat_cat_score: float

    num_num_details: dict[str, dict[tuple[str, str], PairMetricResult]]
    num_cat_details: dict[str, dict[tuple[str, str], PairMetricResult]]
    cat_cat_details: dict[str, dict[tuple[str, str], PairMetricResult]]

    n_pairs_computed: int

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """

        def convert_details(
            details: dict[str, dict[tuple[str, str], PairMetricResult]],
        ) -> dict[str, dict[str, dict[str, Any]]]:
            return {
                metric: {f"{pair[0]}_{pair[1]}": result.to_dict() for pair, result in pairs.items()}
                for metric, pairs in details.items()
            }

        return {
            "score": self.score,
            "num_num_score": self.num_num_score,
            "num_cat_score": self.num_cat_score,
            "cat_cat_score": self.cat_cat_score,
            "num_num_details": convert_details(self.num_num_details),
            "num_cat_details": convert_details(self.num_cat_details),
            "cat_cat_details": convert_details(self.cat_cat_details),
            "n_pairs_computed": self.n_pairs_computed,
        }

    def get_report_data(self) -> dict[str, Any]:
        """
        Get structured data for report generation.

        Returns:
            Dictionary with all reporting data
        """
        return {
            "metric_id": "fidelity.conditional",
            "score": self.score,
            "interpretation": self._interpret_score(),
            "category_scores": {
                "num_num": self.num_num_score,
                "num_cat": self.num_cat_score,
                "cat_cat": self.cat_cat_score,
            },
            "n_pairs_computed": self.n_pairs_computed,
            "worst_pairs": self._get_worst_pairs(5),
        }

    def _interpret_score(self) -> str:
        """Provide human-readable interpretation of the score."""
        if self.score >= 0.9:
            return "Excelente - Las relaciones entre variables están muy bien preservadas"
        if self.score >= 0.7:
            return "Bueno - Las relaciones entre variables están razonablemente preservadas"
        if self.score >= 0.5:
            return "Moderado - Algunas diferencias en las relaciones entre variables"
        if self.score >= 0.3:
            return "Pobre - Diferencias significativas en las relaciones entre variables"
        return "Muy pobre - Las relaciones entre variables están muy deterioradas"

    def _get_worst_pairs(self, n: int = 5) -> list[dict[str, Any]]:
        """Get the n pairs with worst scores."""
        all_pairs = []

        for category, details in [
            ("num_num", self.num_num_details),
            ("num_cat", self.num_cat_details),
            ("cat_cat", self.cat_cat_details),
        ]:
            for metric_name, pairs in details.items():
                for (col1, col2), result in pairs.items():
                    if result.is_valid:
                        all_pairs.append(
                            {
                                "col1": col1,
                                "col2": col2,
                                "category": category,
                                "metric": metric_name,
                                "score": result.normalized_value,
                            }
                        )

        sorted_pairs = sorted(all_pairs, key=lambda x: x["score"])
        return sorted_pairs[:n]

    @classmethod
    def empty(cls) -> "ConditionalResult":
        """
        Create an empty result when no data is available.

        Returns:
            Empty ConditionalResult instance
        """
        return cls(
            score=1.0,
            num_num_score=1.0,
            num_cat_score=1.0,
            cat_cat_score=1.0,
            num_num_details={},
            num_cat_details={},
            cat_cat_details={},
            n_pairs_computed=0,
        )
