"""
Fidelity Aggregator - Top-level aggregation of all fidelity metrics.

Combines three main categories:
1. Global: Structural metrics (outliers, correlation matrices, MMD, energy distance)
2. Marginal: Per-column distribution metrics (tails, scales, coverage)
3. Conditional: Bivariate relationship metrics (Num↔Num, Num↔Cat, Cat↔Cat)

Supports configurable aggregation at 3 levels:
- Level 1: Metrics → Subcategories
- Level 2: Subcategories → Categories
- Level 3: Categories → Family score
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..aggregation.stochastic_dominance import (
    AggregationLevel,
    hierarchical_aggregate,
    weighted_hierarchical_aggregate,
)
from .conditional import ConditionalAggregator, ConditionalResult
from .marginal import MarginalAggregator, MarginalResult


@dataclass
class GlobalResult:
    """Placeholder for global metrics result."""

    score: float
    details: dict[str, float]


@dataclass
class FidelityResult:
    """Result of complete fidelity assessment."""

    score: float  # Final fidelity score Q ∈ [0, 1]

    # Category scores
    global_score: float
    marginal_score: float
    conditional_score: float

    # Detailed results
    global_result: GlobalResult | None
    marginal_result: MarginalResult | None
    conditional_result: ConditionalResult | None

    # Weights used
    weights: dict[str, float]

    # Summary
    categories_computed: list[str]


class FidelityAggregator:
    """
    Top-level aggregator for fidelity metrics.

    Computes and aggregates:
    - Global metrics: Structural similarity measures
    - Marginal metrics: Per-column distribution fidelity
    - Conditional metrics: Bivariate relationship preservation

    Usage:
        aggregator = FidelityAggregator()
        aggregator.fit(real_df, synth_df)
        result = aggregator.compute()
        print(f"Fidelity Score: {result.score:.4f}")
    """

    def __init__(
        self,
        use_global: bool = True,
        use_marginal: bool = True,
        use_conditional: bool = True,
        weights: dict[str, float] | None = None,
        agg_func_level_1: Callable[[list[float]], float] | None = None,
        agg_func_level_2: Callable[[list[float]], float] | None = None,
        agg_func_level_3: Callable[[list[float]], float] | None = None,
    ):
        """
        Initialize fidelity aggregator with configurable 3-level aggregation.

        Args:
            use_global: Include global metrics
            use_marginal: Include marginal metrics
            use_conditional: Include conditional metrics
            weights: Category weights. Default: equal weights.
                    Keys: 'global', 'marginal', 'conditional'
            agg_func_level_1: Aggregation function for Level 1 (metrics → subcategories)
                             Passed to MarginalAggregator and ConditionalAggregator
            agg_func_level_2: Aggregation function for Level 2 (subcategories → categories)
                             Used in this class for aggregating subcategories
            agg_func_level_3: Aggregation function for Level 3 (categories → family score)
                             Used in this class for final aggregation
        """
        self.use_global = use_global
        self.use_marginal = use_marginal
        self.use_conditional = use_conditional

        # Default weights
        if weights is None:
            weights = {"global": 1.0, "marginal": 1.0, "conditional": 1.0}
        # Validate: non-negative, finite, not all zero
        for k, v in weights.items():
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"weight for category '{k}' must be finite and >= 0 (got {v})")
        if sum(weights.values()) <= 0:
            raise ValueError("sum of category weights must be > 0")
        self.weights = weights

        # Configurable aggregation functions
        self.agg_func_level_1 = agg_func_level_1 or (
            lambda scores: hierarchical_aggregate(scores, AggregationLevel.LEVEL_1_METRICS)
        )
        self.agg_func_level_2 = agg_func_level_2 or (
            lambda scores: hierarchical_aggregate(scores, AggregationLevel.LEVEL_2_SUBCATEGORY)
        )
        self.agg_func_level_3 = agg_func_level_3 or (
            lambda scores: hierarchical_aggregate(scores, AggregationLevel.LEVEL_3_CATEGORY)
        )

        self._marginal_aggregator: MarginalAggregator | None = None
        self.logger = logging.getLogger(__name__)
        self._conditional_aggregator: ConditionalAggregator | None = None

        self._real_data: pd.DataFrame | None = None
        self._synth_data: pd.DataFrame | None = None
        self._result: FidelityResult | None = None

    def fit(self, real_data: pd.DataFrame, synth_data: pd.DataFrame) -> "FidelityAggregator":
        """Initialize with data."""
        self._real_data = real_data
        self._synth_data = synth_data

        if self.use_marginal:
            # Pass level_1 aggregator to MarginalAggregator
            self._marginal_aggregator = MarginalAggregator(agg_func=self.agg_func_level_1)
            self._marginal_aggregator.fit(real_data, synth_data)

        if self.use_conditional:
            # Pass level_1 aggregator to ConditionalAggregator
            self._conditional_aggregator = ConditionalAggregator(agg_func=self.agg_func_level_1)

        return self

    def _compute_global(self) -> GlobalResult:
        """
        Compute global structural metrics.

        TODO: Implement full global metrics:
        - Outlier comparison
        - Correlation matrix similarity
        - MMD (Maximum Mean Discrepancy)
        - Energy Distance
        """
        if self._real_data is None:
            return GlobalResult(score=1.0, details={})

        details = {}

        # Simple correlation matrix comparison
        try:
            real_num = self._real_data.select_dtypes(include=[np.number])
            synth_num = self._synth_data.select_dtypes(include=[np.number])

            if len(real_num.columns) > 1:
                real_corr = real_num.corr().values
                synth_corr = synth_num.corr().values

                # Frobenius norm of difference (normalized)
                corr_diff = np.linalg.norm(real_corr - synth_corr, "fro")
                max_diff = np.sqrt(2 * real_corr.shape[0] ** 2)  # Max possible
                corr_score = 1.0 - min(corr_diff / max_diff, 1.0)

                details["correlation_matrix"] = corr_score
        except Exception as e:
            self.logger.warning("correlation_matrix fallback to 1.0: %s", e)
            details["correlation_matrix"] = 1.0

        # Overall score
        if details:
            score = float(np.mean(list(details.values())))
        else:
            score = 1.0

        return GlobalResult(score=score, details=details)

    def compute(self) -> FidelityResult:
        """Compute all fidelity metrics and aggregate."""
        if self._real_data is None:
            raise ValueError("Must call fit() before compute()")

        category_scores = {}

        global_result: GlobalResult | None = None
        marginal_result: MarginalResult | None = None
        conditional_result: ConditionalResult | None = None

        # Global
        if self.use_global:
            global_result = self._compute_global()
            category_scores["global"] = global_result.score

        # Marginal
        if self.use_marginal and self._marginal_aggregator:
            marginal_result = self._marginal_aggregator.compute()
            category_scores["marginal"] = marginal_result.score

        # Conditional
        if self.use_conditional and self._conditional_aggregator:
            conditional_result = self._conditional_aggregator.compute(
                self._real_data, self._synth_data
            )
            category_scores["conditional"] = conditional_result.score

        # Level 3: Aggregate categories → final fidelity score
        if category_scores:
            cat_keys = list(category_scores.keys())
            cat_values = [category_scores[k] for k in cat_keys]
            weight_vec = np.array([self.weights.get(k, 1.0) for k in cat_keys], dtype=float)
            uniform = np.allclose(weight_vec, weight_vec[0])
            if uniform:
                # Preserve user-provided agg_func when weights are uniform
                final_score = float(self.agg_func_level_3(cat_values))
            else:
                final_score = float(
                    weighted_hierarchical_aggregate(
                        np.asarray(cat_values, dtype=float),
                        weight_vec,
                        AggregationLevel.LEVEL_3_CATEGORY,
                    )
                )
        else:
            final_score = 0.0

        self._result = FidelityResult(
            score=final_score,
            global_score=category_scores.get("global", 0.0),
            marginal_score=category_scores.get("marginal", 0.0),
            conditional_score=category_scores.get("conditional", 0.0),
            global_result=global_result,
            marginal_result=marginal_result,
            conditional_result=conditional_result,
            weights=self.weights,
            categories_computed=list(category_scores.keys()),
        )

        return self._result

    def get_category_breakdown(self) -> dict[str, float]:
        """Get scores for each category."""
        if self._result is None:
            raise ValueError("Must call compute() first")

        return {
            "global": self._result.global_score,
            "marginal": self._result.marginal_score,
            "conditional": self._result.conditional_score,
        }

    def get_detailed_report(self) -> dict:
        """Get comprehensive report with all details."""
        if self._result is None:
            raise ValueError("Must call compute() first")

        report = {
            "overall_score": self._result.score,
            "category_scores": self.get_category_breakdown(),
            "weights": self._result.weights,
        }

        if self._result.marginal_result:
            report["marginal_details"] = {
                "tails": self._result.marginal_result.tails_score,
                "scales": self._result.marginal_result.scales_score,
                "coverage": self._result.marginal_result.coverage_score,
                "worst_columns": sorted(
                    self._result.marginal_result.column_scores.items(),
                    key=lambda x: x[1],
                )[:5],
            }

        if self._result.conditional_result:
            report["conditional_details"] = {
                "num_num": self._result.conditional_result.num_num_score,
                "num_cat": self._result.conditional_result.num_cat_score,
                "cat_cat": self._result.conditional_result.cat_cat_score,
                "n_pairs": self._result.conditional_result.n_pairs_computed,
            }

        return report
