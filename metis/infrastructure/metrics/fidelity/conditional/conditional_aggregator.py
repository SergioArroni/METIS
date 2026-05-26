"""
Conditional metrics aggregator.

Aggregates results from all conditional metric categories
(Num↔Num, Num↔Cat, Cat↔Cat) into a single score with configurable Level 1 aggregation.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from .cat_cat import CatCatMetrics
from .conditional_result import ConditionalResult
from .num_cat import NumCatMetrics
from .num_num import NumNumMetrics


class ConditionalAggregator:
    """
    Aggregator for conditional/bivariate relationship metrics with configurable Level 1 aggregation.

    Computes how well synthetic data preserves relationships between
    pairs of columns across three categories:
    - Num↔Num: Correlation preservation
    - Num↔Cat: Association strength preservation
    - Cat↔Cat: Contingency table preservation

    The final score is aggregated using the provided aggregation function.

    Attributes:
        use_num_num: Whether to compute Num↔Num metrics
        use_num_cat: Whether to compute Num↔Cat metrics
        use_cat_cat: Whether to compute Cat↔Cat metrics
        max_pairs: Maximum number of pairs to compute per category
        agg_func: Aggregation function for Level 1 (metrics → subcategories)

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame(
        ...     {"age": [25, 30, 35], "income": [50000, 60000, 70000], "gender": ["M", "F", "M"]}
        ... )
        >>> synth = pd.DataFrame(
        ...     {"age": [26, 31, 34], "income": [51000, 59000, 71000], "gender": ["M", "F", "M"]}
        ... )
        >>> aggregator = ConditionalAggregator(agg_func=mean_func)
        >>> result = aggregator.compute(real, synth)
        >>> 0 <= result.score <= 1
        True
    """

    def __init__(
        self,
        use_num_num: bool = True,
        use_num_cat: bool = True,
        use_cat_cat: bool = True,
        max_pairs: int | None = 100,
        agg_func: Callable[[list[float]], float] | None = None,
    ):
        """
        Initialize the aggregator with configurable Level 1 aggregation.

        Args:
            use_num_num: Whether to compute Num↔Num metrics
            use_num_cat: Whether to compute Num↔Cat metrics
            use_cat_cat: Whether to compute Cat↔Cat metrics
            max_pairs: Maximum number of pairs to compute per category
            agg_func: Aggregation function for Level 1 (metrics within each pair type).
                     If None, uses mean
        """
        self.use_num_num = use_num_num
        self.use_num_cat = use_num_cat
        self.use_cat_cat = use_cat_cat
        self.max_pairs = max_pairs
        self.agg_func = agg_func or np.mean

        self._result: ConditionalResult | None = None

    def compute(self, real_data: pd.DataFrame, synth_data: pd.DataFrame) -> ConditionalResult:
        """
        Compute all conditional metrics.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset

        Returns:
            ConditionalResult with aggregated scores and detailed results
        """
        num_num_results = {}
        num_cat_results = {}
        cat_cat_results = {}

        if self.use_num_num:
            num_num = NumNumMetrics()
            num_num_results = num_num.compute(real_data, synth_data)

        if self.use_num_cat:
            num_cat = NumCatMetrics()
            num_cat_results = num_cat.compute(real_data, synth_data)

        if self.use_cat_cat:
            cat_cat = CatCatMetrics()
            cat_cat_results = cat_cat.compute(real_data, synth_data)

        # Aggregate scores using configurable function
        num_num_score = self._aggregate_category(num_num_results)
        num_cat_score = self._aggregate_category(num_cat_results)
        cat_cat_score = self._aggregate_category(cat_cat_results)

        # Final score: aggregate active categories using configurable function
        active_scores = []
        if self.use_num_num:
            active_scores.append(num_num_score)
        if self.use_num_cat:
            active_scores.append(num_cat_score)
        if self.use_cat_cat:
            active_scores.append(cat_cat_score)

        final_score = float(self.agg_func(active_scores)) if active_scores else 1.0

        # Count total pairs
        n_pairs = sum(
            len(m)
            for results in [num_num_results, num_cat_results, cat_cat_results]
            for m in results.values()
        )

        self._result = ConditionalResult(
            score=final_score,
            num_num_score=num_num_score,
            num_cat_score=num_cat_score,
            cat_cat_score=cat_cat_score,
            num_num_details=num_num_results,
            num_cat_details=num_cat_results,
            cat_cat_details=cat_cat_results,
            n_pairs_computed=n_pairs,
        )

        return self._result

    def _aggregate_category(self, results: dict) -> float:
        """
        Aggregate scores from a category of metrics using configurable function.

        Args:
            results: Dictionary of metric results

        Returns:
            Aggregated score in [0, 1]
        """
        all_scores = []
        for metric_results in results.values():
            for pair_result in metric_results.values():
                if pair_result.is_valid:
                    all_scores.append(pair_result.normalized_value)

        if not all_scores:
            return 1.0  # No pairs = perfect

        # Use configurable aggregation function
        return float(self.agg_func(all_scores))

    @property
    def result(self) -> ConditionalResult | None:
        """Get the last computed result."""
        return self._result
