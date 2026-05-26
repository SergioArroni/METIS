"""
Privacy Aggregator - Top-level aggregation of all privacy metrics.

Combines two main categories:
1. Dataset-based: Metrics computed from data comparison
   - Attribute Inference: MIA, Inference Attack
   - Reidentification: k-Anonymity, l-Diversity, t-Closeness, Record Linkage
   - Empirical Similarity: DCR, NNAA

2. Mechanism-based: Metrics requiring generative model access
   - Differential Privacy

Uses hierarchical Stochastic Dominance:
- FSD for Levels 1-2 (metrics → subcategories → categories)
- SSD for Levels 3-4 (categories → final privacy score)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..aggregation.stochastic_dominance import AggregationLevel, hierarchical_aggregate

# Import all privacy metrics
from .dataset_based.attribute_inference import InferenceAttackMetric, MembershipInferenceMetric
from .dataset_based.empirical_similarity import DCRMetric, NNAAMetric
from .dataset_based.reidentification import (
    KAnonymityMetric,
    LDiversityMetric,
    RecordLinkageMetric,
    TClosenessMetric,
)
from .mechanism_based import DifferentialPrivacyMetric


@dataclass
class SubcategoryResult:
    """Result for a privacy subcategory."""

    name: str
    score: float
    metric_scores: dict[str, float]
    metric_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryResult:
    """Result for a privacy category (Dataset-based or Mechanism-based)."""

    name: str
    score: float
    subcategory_scores: dict[str, float]
    subcategory_results: dict[str, SubcategoryResult] = field(default_factory=dict)


@dataclass
class PrivacyResult:
    """Result of complete privacy assessment."""

    score: float  # Final privacy score Q ∈ [0, 1]

    # Category scores
    dataset_based_score: float
    mechanism_based_score: float

    # Detailed results
    dataset_based_result: CategoryResult | None
    mechanism_based_result: CategoryResult | None

    # Summary
    categories_computed: list[str]
    n_metrics_computed: int

    # Aggregation info
    aggregation_method: str = "hierarchical_fsd_ssd"


class PrivacyAggregator:
    """
    Top-level aggregator for privacy metrics.

    Computes and aggregates:
    - Dataset-based metrics: Attribute Inference, Reidentification, Empirical Similarity
    - Mechanism-based metrics: Differential Privacy

    Uses hierarchical aggregation:
    - Level 1: Individual metrics → Subcategory score (FSD)
    - Level 2: Subcategories → Category score (FSD)
    - Level 3: Categories → Final privacy score (SSD)

    Usage:
        aggregator = PrivacyAggregator()
        aggregator.fit(real_df, synth_df, context)
        result = aggregator.compute()
        print(f"Privacy Score: {result.score:.4f}")
    """

    def __init__(
        self,
        use_dataset_based: bool = True,
        use_mechanism_based: bool = True,
        use_attribute_inference: bool = True,
        use_reidentification: bool = True,
        use_empirical_similarity: bool = True,
        agg_func_level_1: Any | None = None,
        agg_func_level_2: Any | None = None,
        agg_func_level_3: Any | None = None,
    ):
        """
        Initialize privacy aggregator.

        Args:
            use_dataset_based: Include dataset-based metrics
            use_mechanism_based: Include mechanism-based metrics
            use_attribute_inference: Include MIA and Inference Attack
            use_reidentification: Include k-Anonymity, l-Diversity, etc.
            use_empirical_similarity: Include DCR, NNAA
            agg_func_level_1: Aggregation function for Level 1 (metrics → subcategories)
            agg_func_level_2: Aggregation function for Level 2 (subcategories → categories)
            agg_func_level_3: Aggregation function for Level 3 (categories → final score)
        """
        self.use_dataset_based = use_dataset_based
        self.use_mechanism_based = use_mechanism_based
        self.use_attribute_inference = use_attribute_inference
        self.use_reidentification = use_reidentification
        self.use_empirical_similarity = use_empirical_similarity

        # Configurable aggregation functions
        # Default to hierarchical_aggregate with appropriate levels if not provided
        self.agg_func_level_1 = agg_func_level_1 or (
            lambda scores: hierarchical_aggregate(
                np.array(scores), AggregationLevel.LEVEL_1_METRICS
            )
        )
        self.agg_func_level_2 = agg_func_level_2 or (
            lambda scores: hierarchical_aggregate(
                np.array(scores), AggregationLevel.LEVEL_2_SUBCATEGORY
            )
        )
        self.agg_func_level_3 = agg_func_level_3 or (
            lambda scores: hierarchical_aggregate(
                np.array(scores), AggregationLevel.LEVEL_3_CATEGORY
            )
        )

        self._real_data: pd.DataFrame | None = None
        self._synth_data: pd.DataFrame | None = None
        self._context: dict[str, Any] = {}
        self._result: PrivacyResult | None = None
        self.logger = logging.getLogger(__name__)

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> "PrivacyAggregator":
        """Initialize with data and context."""
        self._real_data = real_data
        self._synth_data = synth_data
        self._context = context or {}
        return self

    def _filter_valid_scores(
        self, metric_scores: dict[str, float], metric_details: dict[str, Any]
    ) -> dict[str, float]:
        """
        Filter out scores where metric failed (has 'error' in details).

        Privacy metrics use _create_error_result() which returns value=NaN
        when they fail internally. We also exclude these from aggregation
        to avoid contaminating the aggregate score.

        Args:
            metric_scores: dict of metric names to scores
            metric_details: dict of metric names to details (may contain 'error')

        Returns:
            Filtered dict with only valid scores (no errors in details)
        """
        valid_scores = {}
        for metric_name, score in metric_scores.items():
            details = metric_details.get(metric_name, {})
            # Exclude if details contains 'error' key
            if "error" in details:
                error_msg = details["error"]
                self.logger.warning(
                    f"Privacy metric '{metric_name}' failed: {error_msg}. "
                    f"Excluding from aggregation."
                )
            else:
                valid_scores[metric_name] = score
        return valid_scores

    def _compute_attribute_inference(self) -> SubcategoryResult:
        """Compute attribute inference subcategory."""
        metric_scores = {}
        metric_details = {}

        # MIA
        try:
            mia = MembershipInferenceMetric()
            mia.fit(self._real_data, self._synth_data, self._context)
            mia_result = mia.compute()
            metric_scores["mia"] = mia_result.value
            metric_details["mia"] = mia_result.details
        except Exception as e:
            metric_details["mia"] = {"error": str(e)}

        # Inference Attack
        try:
            inference = InferenceAttackMetric()
            inference.fit(self._real_data, self._synth_data, self._context)
            inference_result = inference.compute()
            metric_scores["inference_attack"] = inference_result.value
            metric_details["inference_attack"] = inference_result.details
        except Exception as e:
            metric_details["inference_attack"] = {"error": str(e)}

        # Filter out failed metrics (those with 'error' in details)
        valid_scores = self._filter_valid_scores(metric_scores, metric_details)

        # Aggregate with configurable function (Level 1) - only valid scores
        if valid_scores:
            scores_list = list(valid_scores.values())
            subcategory_score = float(self.agg_func_level_1(scores_list))
        else:
            subcategory_score = 0.0

        return SubcategoryResult(
            name="attribute_inference",
            score=subcategory_score,
            metric_scores=metric_scores,  # Keep all for reporting
            metric_details=metric_details,
        )

    def _compute_reidentification(self) -> SubcategoryResult:
        """Compute reidentification subcategory."""
        metric_scores = {}
        metric_details = {}

        # k-Anonymity
        try:
            k_anon = KAnonymityMetric()
            k_anon.fit(self._real_data, self._synth_data, self._context)
            k_anon_result = k_anon.compute()
            metric_scores["k_anonymity"] = k_anon_result.value
            metric_details["k_anonymity"] = k_anon_result.details
        except Exception as e:
            metric_details["k_anonymity"] = {"error": str(e)}

        # l-Diversity
        try:
            l_div = LDiversityMetric()
            l_div.fit(self._real_data, self._synth_data, self._context)
            l_div_result = l_div.compute()
            metric_scores["l_diversity"] = l_div_result.value
            metric_details["l_diversity"] = l_div_result.details
        except Exception as e:
            metric_details["l_diversity"] = {"error": str(e)}

        # t-Closeness
        try:
            t_close = TClosenessMetric()
            t_close.fit(self._real_data, self._synth_data, self._context)
            t_close_result = t_close.compute()
            metric_scores["t_closeness"] = t_close_result.value
            metric_details["t_closeness"] = t_close_result.details
        except Exception as e:
            metric_details["t_closeness"] = {"error": str(e)}

        # Record Linkage
        try:
            record_link = RecordLinkageMetric()
            record_link.fit(self._real_data, self._synth_data, self._context)
            record_link_result = record_link.compute()
            metric_scores["record_linkage"] = record_link_result.value
            metric_details["record_linkage"] = record_link_result.details
        except Exception as e:
            metric_details["record_linkage"] = {"error": str(e)}

        # Filter out failed metrics (those with 'error' in details)
        valid_scores = self._filter_valid_scores(metric_scores, metric_details)

        # Aggregate with configurable function (Level 1) - only valid scores
        if valid_scores:
            scores_list = list(valid_scores.values())
            subcategory_score = float(self.agg_func_level_1(scores_list))
        else:
            subcategory_score = 0.0

        return SubcategoryResult(
            name="reidentification",
            score=subcategory_score,
            metric_scores=metric_scores,  # Keep all for reporting
            metric_details=metric_details,
        )

    def _compute_empirical_similarity(self) -> SubcategoryResult:
        """Compute empirical similarity subcategory."""
        metric_scores = {}
        metric_details = {}

        # DCR
        try:
            dcr = DCRMetric()
            dcr.fit(self._real_data, self._synth_data, self._context)
            dcr_result = dcr.compute()
            metric_scores["dcr"] = dcr_result.value
            metric_details["dcr"] = dcr_result.details
        except Exception as e:
            metric_details["dcr"] = {"error": str(e)}

        # NNAA
        try:
            nnaa = NNAAMetric()
            nnaa.fit(self._real_data, self._synth_data, self._context)
            nnaa_result = nnaa.compute()
            metric_scores["nnaa"] = nnaa_result.value
            metric_details["nnaa"] = nnaa_result.details
        except Exception as e:
            metric_details["nnaa"] = {"error": str(e)}

        # Filter out failed metrics (those with 'error' in details)
        valid_scores = self._filter_valid_scores(metric_scores, metric_details)

        # Aggregate with configurable function (Level 1) - only valid scores
        if valid_scores:
            scores_list = list(valid_scores.values())
            subcategory_score = float(self.agg_func_level_1(scores_list))
        else:
            subcategory_score = 0.0

        return SubcategoryResult(
            name="empirical_similarity",
            score=subcategory_score,
            metric_scores=metric_scores,  # Keep all for reporting
            metric_details=metric_details,
        )

    def _compute_dataset_based(self) -> CategoryResult:
        """Compute all dataset-based metrics."""
        subcategory_scores = {}
        subcategory_results = {}

        # Attribute Inference
        if self.use_attribute_inference:
            attr_inf = self._compute_attribute_inference()
            subcategory_scores["attribute_inference"] = attr_inf.score
            subcategory_results["attribute_inference"] = attr_inf

        # Reidentification
        if self.use_reidentification:
            reident = self._compute_reidentification()
            subcategory_scores["reidentification"] = reident.score
            subcategory_results["reidentification"] = reident

        # Empirical Similarity
        if self.use_empirical_similarity:
            emp_sim = self._compute_empirical_similarity()
            subcategory_scores["empirical_similarity"] = emp_sim.score
            subcategory_results["empirical_similarity"] = emp_sim

        # Filter out subcategories where all metrics failed (score=0.0 with no valid metrics)
        valid_subcategory_scores = {}
        for subcat_name, score in subcategory_scores.items():
            result = subcategory_results[subcat_name]
            # Check if at least one metric succeeded (no error in details)
            has_valid_metric = any(
                "error" not in result.metric_details.get(metric_name, {})
                for metric_name in result.metric_scores
            )
            if has_valid_metric:
                valid_subcategory_scores[subcat_name] = score

        # Aggregate subcategories with configurable function (Level 2) - only valid ones
        if valid_subcategory_scores:
            scores_list = list(valid_subcategory_scores.values())
            category_score = float(self.agg_func_level_2(scores_list))
        else:
            category_score = 0.0

        return CategoryResult(
            name="dataset_based",
            score=category_score,
            subcategory_scores=subcategory_scores,  # Keep all for reporting
            subcategory_results=subcategory_results,
        )

    def _compute_mechanism_based(self) -> CategoryResult:
        """Compute all mechanism-based metrics."""
        metric_scores = {}
        metric_details = {}

        # Differential Privacy
        try:
            dp = DifferentialPrivacyMetric()
            dp.fit(self._real_data, self._synth_data, self._context)
            dp_result = dp.compute()
            metric_scores["differential_privacy"] = dp_result.value
            metric_details["differential_privacy"] = dp_result.details
        except Exception as e:
            metric_details["differential_privacy"] = {"error": str(e)}

        # Filter out failed metrics (those with 'error' in details)
        valid_scores = self._filter_valid_scores(metric_scores, metric_details)

        # For mechanism-based, we only have one subcategory (DP)
        # So subcategory_score = category_score
        if valid_scores:
            scores_list = list(valid_scores.values())
            category_score = float(self.agg_func_level_1(scores_list))
        else:
            category_score = 0.0

        # Wrap in subcategory result
        dp_subcategory = SubcategoryResult(
            name="differential_privacy",
            score=category_score,
            metric_scores=metric_scores,
            metric_details=metric_details,
        )

        return CategoryResult(
            name="mechanism_based",
            score=category_score,
            subcategory_scores={"differential_privacy": category_score},
            subcategory_results={"differential_privacy": dp_subcategory},
        )

    def compute(self) -> PrivacyResult:
        """Compute all privacy metrics and aggregate."""
        if self._real_data is None:
            raise ValueError("Must call fit() before compute()")

        category_scores = {}
        n_metrics = 0

        dataset_based_result: CategoryResult | None = None
        mechanism_based_result: CategoryResult | None = None

        # Dataset-based
        if self.use_dataset_based:
            dataset_based_result = self._compute_dataset_based()
            category_scores["dataset_based"] = dataset_based_result.score

            # Count metrics
            for subcat in dataset_based_result.subcategory_results.values():
                n_metrics += len(subcat.metric_scores)

        # Mechanism-based
        if self.use_mechanism_based:
            mechanism_based_result = self._compute_mechanism_based()
            category_scores["mechanism_based"] = mechanism_based_result.score

            # Count metrics
            for subcat in mechanism_based_result.subcategory_results.values():
                n_metrics += len(subcat.metric_scores)

        # Filter out categories where all metrics failed (score=0.0 with no valid metrics)
        valid_category_scores = {}
        for cat_name, score in category_scores.items():
            if cat_name == "dataset_based" and dataset_based_result:
                # Check if at least one subcategory has valid metrics
                has_valid = any(
                    any(
                        "error" not in subcat.metric_details.get(m, {})
                        for m in subcat.metric_scores
                    )
                    for subcat in dataset_based_result.subcategory_results.values()
                )
                if has_valid:
                    valid_category_scores[cat_name] = score
            elif cat_name == "mechanism_based" and mechanism_based_result:
                # Check if at least one subcategory has valid metrics
                has_valid = any(
                    any(
                        "error" not in subcat.metric_details.get(m, {})
                        for m in subcat.metric_scores
                    )
                    for subcat in mechanism_based_result.subcategory_results.values()
                )
                if has_valid:
                    valid_category_scores[cat_name] = score

        # Aggregate categories with configurable function (Level 3) - only valid ones
        if valid_category_scores:
            scores_list = list(valid_category_scores.values())
            final_score = float(self.agg_func_level_3(scores_list))
        else:
            final_score = 0.0

        self._result = PrivacyResult(
            score=final_score,
            dataset_based_score=category_scores.get("dataset_based", 0.0),
            mechanism_based_score=category_scores.get("mechanism_based", 0.0),
            dataset_based_result=dataset_based_result,
            mechanism_based_result=mechanism_based_result,
            categories_computed=list(category_scores.keys()),  # Keep all for reporting
            n_metrics_computed=n_metrics,
        )

        return self._result

    def get_category_breakdown(self) -> dict[str, float]:
        """Get scores for each category."""
        if self._result is None:
            raise ValueError("Must call compute() first")

        return {
            "dataset_based": self._result.dataset_based_score,
            "mechanism_based": self._result.mechanism_based_score,
        }

    def get_subcategory_breakdown(self) -> dict[str, dict[str, float]]:
        """Get scores for each subcategory within categories."""
        if self._result is None:
            raise ValueError("Must call compute() first")

        breakdown = {}

        if self._result.dataset_based_result:
            breakdown["dataset_based"] = self._result.dataset_based_result.subcategory_scores

        if self._result.mechanism_based_result:
            breakdown["mechanism_based"] = self._result.mechanism_based_result.subcategory_scores

        return breakdown

    def get_detailed_report(self) -> dict:
        """Get comprehensive report with all details."""
        if self._result is None:
            raise ValueError("Must call compute() first")

        report = {
            "overall_score": self._result.score,
            "category_scores": self.get_category_breakdown(),
            "subcategory_scores": self.get_subcategory_breakdown(),
            "n_metrics_computed": self._result.n_metrics_computed,
            "aggregation_method": self._result.aggregation_method,
        }

        # Add metric-level details
        metric_details = {}

        if self._result.dataset_based_result:
            for (
                subcat_name,
                subcat,
            ) in self._result.dataset_based_result.subcategory_results.items():
                for metric_name, score in subcat.metric_scores.items():
                    metric_details[f"dataset_based.{subcat_name}.{metric_name}"] = {
                        "score": score,
                        "details": subcat.metric_details.get(metric_name, {}),
                    }

        if self._result.mechanism_based_result:
            for (
                subcat_name,
                subcat,
            ) in self._result.mechanism_based_result.subcategory_results.items():
                for metric_name, score in subcat.metric_scores.items():
                    metric_details[f"mechanism_based.{subcat_name}.{metric_name}"] = {
                        "score": score,
                        "details": subcat.metric_details.get(metric_name, {}),
                    }

        report["metrics"] = metric_details

        return report
