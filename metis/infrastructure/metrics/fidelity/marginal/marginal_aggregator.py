"""
Marginal Aggregator - Combines Tails, Scales, and Coverage into single marginal score.

The marginal fidelity score aggregates three subcategories:
1. Tails: Distribution tail comparison (KS, Wasserstein, Hellinger, etc.)
2. Scales: Central tendency and dispersion (Mean, Median, MAD, IQR, Cohen's d)
3. Coverage: Categorical/coverage metrics (TVD, KL, JS, PSI, Entropy)

Uses configurable aggregation for Level 1 (metrics → subcategories).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from ...aggregation.stochastic_dominance import aggregate_metrics
from .coverage import CoverageAggregator, CoverageResult
from .marginal_result import MarginalResult
from .scales import ScalesAggregator, ScalesResult
from .tails import TailsAggregator, TailsResult


class MarginalAggregator:
    """
    Aggregator for marginal fidelity metrics with configurable Level 1 aggregation.

    Combines three subcategories:
    - Tails: tail distribution metrics
    - Scales: central tendency and dispersion metrics
    - Coverage: categorical distribution metrics

    Each subcategory produces a score per column. These are aggregated using
    the provided aggregation function (Level 1) or Stochastic Dominance by default.

    Usage:
        aggregator = MarginalAggregator(agg_func=mean_func)
        aggregator.fit(real_df, synth_df)
        result = aggregator.compute()
        print(f"Marginal Score: {result.score:.4f}")
    """

    def __init__(
        self,
        use_tails: bool = True,
        use_scales: bool = True,
        use_coverage: bool = True,
        tails_metrics: list[str] | None = None,
        scales_metrics: list[str] | None = None,
        coverage_metrics: list[str] | None = None,
        agg_func: Callable[[list[float]], float] | None = None,
    ):
        """
        Initialize marginal aggregator with configurable Level 1 aggregation.

        Args:
            use_tails: Include tails metrics
            use_scales: Include scales metrics
            use_coverage: Include coverage metrics
            tails_metrics: Specific tails metrics to use (None = defaults)
            scales_metrics: Specific scales metrics to use (None = defaults)
            coverage_metrics: Specific coverage metrics to use (None = defaults)
            agg_func: Aggregation function for Level 1 (metrics → subcategories).
                     If None, uses aggregate_metrics (Stochastic Dominance)
        """
        self.use_tails = use_tails
        self.use_scales = use_scales
        self.use_coverage = use_coverage
        self.agg_func = agg_func  # Store for potential use in subcategories

        self._tails_aggregator: TailsAggregator | None = None
        self._scales_aggregator: ScalesAggregator | None = None
        self._coverage_aggregator: CoverageAggregator | None = None

        if use_tails:
            self._tails_aggregator = TailsAggregator(metrics=tails_metrics)
        if use_scales:
            self._scales_aggregator = ScalesAggregator(metrics=scales_metrics)
        if use_coverage:
            self._coverage_aggregator = CoverageAggregator(metrics=coverage_metrics)

        self._real_data: pd.DataFrame | None = None
        self._synth_data: pd.DataFrame | None = None
        self._result: MarginalResult | None = None

    def fit(self, real_data: pd.DataFrame, synth_data: pd.DataFrame) -> "MarginalAggregator":
        """Initialize with data."""
        self._real_data = real_data
        self._synth_data = synth_data

        if self._tails_aggregator:
            self._tails_aggregator.fit(real_data, synth_data)
        if self._scales_aggregator:
            self._scales_aggregator.fit(real_data, synth_data)
        if self._coverage_aggregator:
            self._coverage_aggregator.fit(real_data, synth_data)

        return self

    def compute(self) -> MarginalResult:
        """Compute all subcategory scores and aggregate."""
        if self._real_data is None:
            raise ValueError("Must call fit() before compute()")

        # Compute each subcategory
        tails_result: TailsResult | None = None
        scales_result: ScalesResult | None = None
        coverage_result: CoverageResult | None = None

        subcategory_scores = {}
        subcategory_column_scores: dict[str, dict[str, float]] = {}

        if self._tails_aggregator:
            tails_result = self._tails_aggregator.compute()
            subcategory_scores["tails"] = tails_result.score
            subcategory_column_scores["tails"] = tails_result.column_scores

        if self._scales_aggregator:
            scales_result = self._scales_aggregator.compute()
            subcategory_scores["scales"] = scales_result.score
            subcategory_column_scores["scales"] = scales_result.column_scores

        if self._coverage_aggregator:
            coverage_result = self._coverage_aggregator.compute()
            subcategory_scores["coverage"] = coverage_result.score
            subcategory_column_scores["coverage"] = coverage_result.column_scores

        # Collect all columns across subcategories
        all_columns = set()
        for subcat_cols in subcategory_column_scores.values():
            all_columns.update(subcat_cols.keys())
        all_columns = sorted(all_columns)

        n_cols = len(all_columns)
        n_subcats = len(subcategory_scores)

        if n_cols == 0 or n_subcats == 0:
            self._result = MarginalResult(
                score=0.0,
                tails_score=subcategory_scores.get("tails", 0.0),
                scales_score=subcategory_scores.get("scales", 0.0),
                coverage_score=subcategory_scores.get("coverage", 0.0),
                tails_result=tails_result,
                scales_result=scales_result,
                coverage_result=coverage_result,
                column_scores={},
                subcategories_used=list(subcategory_scores.keys()),
                n_columns=0,
            )
            return self._result

        # Build matrix A[column, subcategory]
        subcat_names = list(subcategory_scores.keys())
        A = np.zeros((n_cols, n_subcats))

        for i, col in enumerate(all_columns):
            for j, subcat in enumerate(subcat_names):
                if col in subcategory_column_scores[subcat]:
                    A[i, j] = subcategory_column_scores[subcat][col]
                else:
                    # Column not applicable to this subcategory - use subcategory average
                    A[i, j] = subcategory_scores[subcat]

        # Aggregate using Stochastic Dominance (Level 1 aggregation happens within subcategories)
        # This is Level 2: subcategories → category
        column_scores_array, final_score = aggregate_metrics(A)
        column_scores = {col: float(column_scores_array[i]) for i, col in enumerate(all_columns)}

        self._result = MarginalResult(
            score=float(final_score),
            tails_score=subcategory_scores.get("tails", 0.0),
            scales_score=subcategory_scores.get("scales", 0.0),
            coverage_score=subcategory_scores.get("coverage", 0.0),
            tails_result=tails_result,
            scales_result=scales_result,
            coverage_result=coverage_result,
            column_scores=column_scores,
            subcategories_used=subcat_names,
            n_columns=n_cols,
        )

        return self._result

    def get_subcategory_breakdown(self) -> dict[str, float]:
        """Get scores for each subcategory."""
        if self._result is None:
            raise ValueError("Must call compute() first")

        return {
            "tails": self._result.tails_score,
            "scales": self._result.scales_score,
            "coverage": self._result.coverage_score,
        }

    def get_worst_columns(self, n: int = 5) -> list[tuple]:
        """Get the n columns with worst scores."""
        if self._result is None:
            raise ValueError("Must call compute() first")

        sorted_cols = sorted(self._result.column_scores.items(), key=lambda x: x[1])
        return sorted_cols[:n]
