"""
Utility Aggregator - Combines ML Efficiency metrics into a unified utility score.

Aggregates the 4 standalone ML efficiency metrics with 3-level aggregation:
- Level 1: Individual strategies (TTS, TSTR, TRTS, TTRS) → ml_efficiency score
- Level 2: ml_efficiency subcategory → ml_efficiency category (prepared for future expansion)
- Level 3: Categories → Final utility score

Each metric computes: 1 - |strategy - TTR|, so values are already normalized [0, 1]
where 1 = perfect utility (strategy matches TTR baseline).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from metis.domain.entities import MetricResult
from metis.shared.aggregation_registry import get_aggregation_function


@dataclass
class UtilityResult:
    """Result of complete utility assessment."""

    score: float  # Final utility score U ∈ [0, 1]

    # Individual ML efficiency scores
    tts_score: float
    tstr_score: float
    trts_score: float
    ttrs_score: float

    # Intermediate aggregations
    ml_efficiency_score: float  # Level 1 aggregation

    # Metadata
    metrics_computed: list[str]
    n_metrics_computed: int
    aggregation_method: str = "configurable_3_level"


class UtilityAggregator:
    """
    Top-level aggregator for utility metrics.

    Computes and aggregates ML Efficiency metrics:
    - TTS: Train on Synthetic, Test on Synthetic
    - TSTR: Train on Synthetic, Test on Real
    - TRTS: Train on Real, Test on Synthetic
    - TTRS: Train on Real, Test on Real (baseline)

    Uses configurable aggregation function (mean, median, etc.)
    that can be optimized during calibration.

    Usage:
        aggregator = UtilityAggregator(agg_func="mean")
        aggregator.fit(real_df, synth_df, context)
        result = aggregator.compute()
        print(f"Utility Score: {result.score:.4f}")
    """

    def __init__(
        self,
        agg_func_level_1: Callable[[list[float]], float] | None = None,
        agg_func_level_2: Callable[[list[float]], float] | None = None,
        agg_func_level_3: Callable[[list[float]], float] | None = None,
        agg_func_name_level_1: str = "mean",
        agg_func_name_level_2: str = "mean",
        agg_func_name_level_3: str = "mean",
    ):
        """
        Initialize utility aggregator with configurable 3-level aggregation.

        Args:
            agg_func_level_1: Aggregation function for Level 1 (strategies → ml_efficiency)
            agg_func_level_2: Aggregation function for Level 2 (subcategories → categories)
            agg_func_level_3: Aggregation function for Level 3 (categories → final score)
            agg_func_name_level_1: Name of level 1 aggregation function for registry lookup
            agg_func_name_level_2: Name of level 2 aggregation function for registry lookup
            agg_func_name_level_3: Name of level 3 aggregation function for registry lookup
        """
        self.agg_func_name_level_1 = agg_func_name_level_1
        self.agg_func_name_level_2 = agg_func_name_level_2
        self.agg_func_name_level_3 = agg_func_name_level_3

        self.agg_func_level_1 = agg_func_level_1 or get_aggregation_function(agg_func_name_level_1)
        self.agg_func_level_2 = agg_func_level_2 or get_aggregation_function(agg_func_name_level_2)
        self.agg_func_level_3 = agg_func_level_3 or get_aggregation_function(agg_func_name_level_3)

        self._real_data: pd.DataFrame | None = None
        self._synth_data: pd.DataFrame | None = None
        self._context: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self._result: UtilityResult | None = None

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> "UtilityAggregator":
        """Initialize with data and context."""
        self._real_data = real_data
        self._synth_data = synth_data
        self._context = context or {}
        return self

    def compute_from_results(self, metric_results: list[MetricResult]) -> UtilityResult:
        """
        Compute utility score from pre-computed metric results.

        Args:
            metric_results: list of MetricResult objects from TTS, TSTR, TRTS, TTRS

        Returns:
            UtilityResult with aggregated score
        """
        # Extract scores by metric name
        scores_dict = {}
        metrics_computed = []

        for result in metric_results:
            # Extract metric name from ID (e.g., "utility.tts" -> "tts")
            metric_name = result.id.split(".")[-1].lower()

            # Only include ML efficiency standalone metrics
            if metric_name in ["tts", "tstr", "trts", "ttrs"]:
                # Check if metric failed
                if "error" in result.details:
                    error_msg = result.details["error"]
                    self.logger.warning(
                        f"Utility metric '{metric_name}' failed: {error_msg}. "
                        f"Excluding from aggregation."
                    )
                else:
                    scores_dict[metric_name] = result.value
                    metrics_computed.append(metric_name)

        # Get scores in consistent order
        tts_score = scores_dict.get("tts", 0.0)
        tstr_score = scores_dict.get("tstr", 0.0)
        trts_score = scores_dict.get("trts", 0.0)
        ttrs_score = scores_dict.get("ttrs", 0.0)

        # Level 1: Aggregate TTS, TSTR, TRTS, TTRS → ml_efficiency score
        valid_scores = [s for s in scores_dict.values() if s > 0.0]

        if valid_scores:
            ml_efficiency_score = float(self.agg_func_level_1(valid_scores))
        else:
            self.logger.warning("No valid utility metrics computed. Returning score=0.0")
            ml_efficiency_score = 0.0

        # Level 2: Aggregate subcategories → categories
        # Currently we only have ml_efficiency, so this is identity
        # Prepared for future expansion with more subcategories
        category_scores = [ml_efficiency_score]
        if category_scores:
            category_score = float(self.agg_func_level_2(category_scores))
        else:
            category_score = 0.0

        # Level 3: Aggregate categories → final utility score
        # Currently we only have one category, so this is identity
        # Prepared for future expansion with more categories
        final_scores = [category_score]
        if final_scores:
            final_score = float(self.agg_func_level_3(final_scores))
        else:
            final_score = 0.0

        self._result = UtilityResult(
            score=final_score,
            tts_score=tts_score,
            tstr_score=tstr_score,
            trts_score=trts_score,
            ttrs_score=ttrs_score,
            ml_efficiency_score=ml_efficiency_score,
            metrics_computed=metrics_computed,
            n_metrics_computed=len(metrics_computed),
            aggregation_method=f"3-level: {self.agg_func_name_level_1}/{self.agg_func_name_level_2}/{self.agg_func_name_level_3}",
        )

        return self._result

    def get_breakdown(self) -> dict[str, float]:
        """Get scores for each ML efficiency metric."""
        if self._result is None:
            raise ValueError("Must call compute_from_results() first")

        return {
            "tts": self._result.tts_score,
            "tstr": self._result.tstr_score,
            "trts": self._result.trts_score,
            "ttrs": self._result.ttrs_score,
            "ml_efficiency": self._result.ml_efficiency_score,
        }
