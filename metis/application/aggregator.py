"""Result aggregation and scoring logic."""

import statistics
from typing import Any

import numpy as np
from scipy import stats

from ..domain.contracts import BoundsStorage
from ..domain.entities import EvalPlan, MetricResult
from ..domain.taxonomy import FAMILIES, group_metrics_by_hierarchy
from ..infrastructure.runtime.logging import get_logger
from ..shared.aggregation_registry import get_aggregation_function


class Aggregator:
    """Aggregates metric results into family scores and composite indices."""

    @staticmethod
    def _is_valid_metric_result(result: MetricResult) -> bool:
        return "error" not in result.details and np.isfinite(result.value)

    @staticmethod
    def _filter_finite_values(values: list[float]) -> list[float]:
        return [float(value) for value in values if np.isfinite(value)]

    def __init__(
        self,
        risk_aversion: float = 5.0,
        calibration_bounds_path: str | None = None,
        optimal_aggregators_path: str | None = None,
        calibration_bounds: BoundsStorage | None = None,
    ):
        self.risk_aversion = risk_aversion
        self.logger = get_logger(__name__)
        self.calibration_bounds: BoundsStorage | None = calibration_bounds
        self.optimal_aggregators = None

        # Load calibration from path if provided and no instance given
        if calibration_bounds_path and not self.calibration_bounds:
            from metis.calibrate import CalibrationBounds

            self.calibration_bounds = CalibrationBounds.load(calibration_bounds_path)

        # Load optimal aggregators if provided
        if optimal_aggregators_path:
            from metis.calibrate import AggregatorTuner

            self.optimal_aggregators = AggregatorTuner.load(optimal_aggregators_path)

        # Extract optimal aggregators from bounds object (auto-tuned)
        if not self.optimal_aggregators and self.calibration_bounds:
            aggs = getattr(self.calibration_bounds, "optimal_aggregators", None)
            if aggs:
                self.optimal_aggregators = aggs
                self.logger.info("Using auto-tuned composite aggregator from calibration bounds")

        # Determine which aggregation functions to use
        self._setup_aggregation_functions()

    def aggregate(
        self,
        results: list[MetricResult],
        plan: EvalPlan,  # noqa: ARG002
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:  # noqa: ARG002
        """
        Aggregate metric results into family scores and hierarchical breakdowns.

        Args:
            results: list of computed metric results
            plan: Evaluation plan used for the run
            weights: Optional weights for family scoring (defaults to equal weighting)

        Returns:
            Dictionary containing family scores, hierarchical scores, and metadata
        """
        aggregates = {}

        # Calculate family scores
        family_scores_dict = {}
        for family in FAMILIES:
            family_results = [r for r in results if r.family == family]
            if family_results:
                family_score = self._calculate_family_score(family_results, family)
                aggregates[f"{family}_score"] = family_score
                aggregates[f"{family}_count"] = len(family_results)
                if np.isfinite(family_score):
                    family_scores_dict[family] = family_score
                else:
                    self.logger.warning(
                        "Family '%s' score is NaN — excluded from composite",
                        family,
                    )

        # Calculate composite score using optimal aggregator (Level 4: Families → Final score)
        if family_scores_dict:
            family_scores_array = list(family_scores_dict.values())
            composite = self._apply_composite_aggregation(family_scores_array)
            aggregates["composite_score"] = float(composite)
            aggregates["composite_families"] = list(family_scores_dict.keys())

        # Calculate hierarchical scores (by category and subcategory)
        hierarchy_scores = self._calculate_hierarchy_scores(results)
        aggregates["hierarchy"] = hierarchy_scores

        # Apply calibration normalization if available
        if self.calibration_bounds:
            aggregates = self._apply_calibration_normalization(aggregates)

        # Add execution metadata
        aggregates["total_metrics"] = len(results)
        aggregates["successful_metrics"] = len(
            [r for r in results if self._is_valid_metric_result(r)]
        )
        aggregates["failed_metrics"] = len(results) - aggregates["successful_metrics"]

        return aggregates

    def _calculate_hierarchy_scores(self, results: list[MetricResult]) -> dict[str, Any]:
        """
        Calculate aggregated scores for each level of the hierarchy.

        Returns:
            Nested dictionary with scores at each hierarchy level:
            {
                "fidelity": {
                    "score": 0.68,
                    "categories": {
                        "marginal": {
                            "score": 0.65,
                            "subcategories": {
                                "tails": {"score": 0.62, "metrics": ["ks", "wasserstein"]},
                                "scales": {"score": 0.68, "metrics": [...]}
                            }
                        },
                        "global": {
                            "score": 0.71,
                            "metrics": ["correlation_matrix", "mmd"]
                        }
                    }
                }
            }
        """
        hierarchy_scores: dict[str, Any] = {}

        # Build result lookup by metric_id
        result_by_id = {r.id: r for r in results}

        # Group metrics by their hierarchy
        metric_ids = [r.id for r in results]
        grouped = group_metrics_by_hierarchy(metric_ids)

        for family, categories in grouped.items():
            family_data: dict[str, Any] = {"categories": {}}
            family_values = []

            for category, subcategories in categories.items():
                category_data: dict[str, Any] = {}
                category_values = []

                for subcategory, metric_ids_list in subcategories.items():
                    # Get values for these metrics
                    subcat_values = []
                    subcat_metrics = []
                    skipped_ids = []
                    for mid in metric_ids_list:
                        if mid in result_by_id:
                            r = result_by_id[mid]
                            if self._is_valid_metric_result(r):
                                subcat_values.append(r.value)
                                # Extract just the metric name (after the family prefix)
                                metric_name = mid.split(".")[-1] if "." in mid else mid
                                subcat_metrics.append(metric_name)
                            else:
                                skipped_ids.append(mid)
                    if skipped_ids:
                        self.logger.warning(
                            "Hierarchy '%s.%s.%s': skipping non-finite metric(s): %s",
                            family,
                            category,
                            subcategory,
                            skipped_ids,
                        )

                    if subcat_values:
                        # Layer 1: Metrics → Subcategories (use layers_1_2 aggregator)
                        subcat_score = self._apply_aggregation(subcat_values, "1_2")
                        category_values.extend(subcat_values)

                        if subcategory == "_direct":
                            # Metrics directly under category (no subcategory)
                            category_data["score"] = subcat_score
                            category_data["metrics"] = subcat_metrics
                        else:
                            if "subcategories" not in category_data:
                                category_data["subcategories"] = {}
                            category_data["subcategories"][subcategory] = {
                                "score": subcat_score,
                                "count": len(subcat_metrics),
                                "metrics": subcat_metrics,
                            }

                # Calculate category score
                # Layer 2: Subcategories → Categories (use layers_1_2 aggregator)
                if category_values:
                    category_data["score"] = self._apply_aggregation(category_values, "1_2")
                    category_data["count"] = len(category_values)
                    family_values.extend(category_values)

                family_data["categories"][category] = category_data

            # Calculate family score
            # Layer 3: Categories → Families (use layers_3_4 aggregator)
            if family_values:
                family_data["score"] = self._apply_aggregation(family_values, "3_4")
                family_data["count"] = len(family_values)

            hierarchy_scores[family] = family_data

        return hierarchy_scores

    def _calculate_family_score(self, family_results: list[MetricResult], family: str) -> float:
        """
        Calculate aggregated score for a metric family.

        Uses the per-family tuned aggregator when available (from
        calibration's ``tune_from_metrics``), otherwise falls back to
        the generic ``agg_func_3_4``.

        Args:
            family_results: Results for metrics in this family
            family: Family name used to look up tuned aggregator

        Returns:
            Aggregated score for the family (0.0 to 1.0)
        """
        if not family_results:
            return float("nan")

        # Filter out failed metrics
        successful_results = [r for r in family_results if self._is_valid_metric_result(r)]
        skipped = [r for r in family_results if not self._is_valid_metric_result(r)]
        if skipped:
            self.logger.warning(
                "Family '%s': skipping %d non-finite/errored metric(s) from aggregation: %s",
                family,
                len(skipped),
                [r.id for r in skipped],
            )
        if not successful_results:
            self.logger.warning(
                "Family '%s': all %d metric(s) are non-finite/errored — family score is NaN",
                family,
                len(family_results),
            )
            return float("nan")

        values = [r.value for r in successful_results]

        # Use family-specific tuned aggregator if available
        if hasattr(self, "_family_agg_funcs") and family in self._family_agg_funcs:
            return float(self._family_agg_funcs[family](values))

        return self._apply_aggregation(values, "3_4")

    def _setup_aggregation_functions(self) -> None:
        """
        Setup aggregation functions based on optimal configuration.

        Supports two formats:
        1. New format (per-family, per-level):
           {
             "fidelity": {"level_1": "mean", "level_2": "median", "level_3": "ssd"},
             "privacy": {"level_1": "mean", "level_2": "median", "level_3": "ssd"},
             "utility": {"level_1": "mean", "level_2": "mean", "level_3": "ssd"},
             "composite": "ssd"
           }
        2. Old format (backward compatibility):
           {
             "layers_1_2": "mean",
             "layers_3_4": "ssd"
           }

        If no configuration is loaded, uses defaults.
        """
        if self.optimal_aggregators:
            # Check format
            if "fidelity" in self.optimal_aggregators and isinstance(
                self.optimal_aggregators.get("fidelity"), dict
            ):
                # NEW FORMAT: Per-family, per-level configuration
                self.logger.info("Using new per-family per-level aggregation configuration")

                # Extract configurations for each family
                self.fidelity_aggs = self.optimal_aggregators.get("fidelity", {})
                self.privacy_aggs = self.optimal_aggregators.get("privacy", {})
                self.utility_aggs = self.optimal_aggregators.get("utility", {})
                self.composite_agg_name = self.optimal_aggregators.get("composite", "ssd")

                # For backward compatibility with hierarchy scoring (old system)
                # Map new format to old layers_1_2 and layers_3_4
                # Use fidelity's level_2 as representative for layers_1_2
                # Use composite for layers_3_4
                self.agg_layers_1_2_name = self.fidelity_aggs.get("level_2", "mean")
                self.agg_layers_3_4_name = self.composite_agg_name

            elif "layers_1_2" in self.optimal_aggregators:
                # OLD FORMAT: Backward compatibility
                self.logger.info(
                    "Using old layers-based aggregation configuration (backward compatibility)"
                )
                self.agg_layers_1_2_name = self.optimal_aggregators.get("layers_1_2", "mean")
                self.agg_layers_3_4_name = self.optimal_aggregators.get("layers_3_4", "ssd")

                # Map old format to new (all families use same aggregators)
                self.fidelity_aggs = {
                    "level_1": self.agg_layers_1_2_name,
                    "level_2": self.agg_layers_1_2_name,
                    "level_3": self.agg_layers_3_4_name,
                }
                self.privacy_aggs = self.fidelity_aggs.copy()
                self.utility_aggs = self.fidelity_aggs.copy()
                self.composite_agg_name = self.agg_layers_3_4_name
            elif all(
                isinstance(self.optimal_aggregators.get(f), str)
                for f in ("fidelity", "privacy", "utility")
                if f in self.optimal_aggregators
            ) and any(f in self.optimal_aggregators for f in ("fidelity", "privacy", "utility")):
                # TUNED PER-FAMILY FORMAT from tune_from_metrics():
                # {"fidelity": "median", "privacy": "mean", ..., "composite": "ssd"}
                self.logger.info(
                    "Using per-family tuned aggregators: %s",
                    dict(self.optimal_aggregators),
                )
                self._use_default_aggregators()  # baseline defaults

                # Build per-family aggregation functions
                self._family_agg_funcs: dict[str, Any] = {}
                for fam in ("fidelity", "privacy", "utility"):
                    agg_name = self.optimal_aggregators.get(fam)
                    if agg_name:
                        self._family_agg_funcs[fam] = self._build_aggregator(agg_name)
                        # Also update the per-level dicts for hierarchy compatibility
                        setattr(
                            self,
                            f"{fam}_aggs",
                            {
                                "level_1": agg_name,
                                "level_2": agg_name,
                                "level_3": agg_name,
                            },
                        )

                self.composite_agg_name = self.optimal_aggregators.get("composite", "ssd")
                # Update layers_3_4 to match the most common family aggregator
                # (for hierarchy scoring backward compat)
                fam_aggs = [
                    self.optimal_aggregators.get(f, "median")
                    for f in ("fidelity", "privacy", "utility")
                ]
                self.agg_layers_3_4_name = max(set(fam_aggs), key=fam_aggs.count)

            # COMPOSITE-ONLY FORMAT: from auto-tuning (only composite aggregator tuned)
            # Per-family stays at defaults, only composite changes
            elif "composite" in self.optimal_aggregators:
                self.logger.info(
                    "Using auto-tuned composite aggregator: %s",
                    self.optimal_aggregators["composite"],
                )
                self._use_default_aggregators()
                self.composite_agg_name = self.optimal_aggregators["composite"]
            else:
                self.logger.warning("Unknown aggregation configuration format. Using defaults.")
                self._use_default_aggregators()
        else:
            # No configuration, use defaults
            self._use_default_aggregators()

        # Build aggregation function objects for old system (hierarchy scoring)
        self.agg_func_1_2 = get_aggregation_function(self.agg_layers_1_2_name)
        self.agg_func_3_4 = self._build_aggregator(self.agg_layers_3_4_name)
        self.agg_func_composite = self._build_aggregator(self.composite_agg_name)

    def _build_aggregator(self, name: str):
        """Resolve an aggregator name into a callable.

        Centralises the special-cased ``ssd`` / ``hypervolume`` handling so
        every callsite uses the same dispatch table (OCP). New aggregator
        kinds only need to be added once here.
        """
        builders: dict[str, Any] = {
            "ssd": lambda: get_aggregation_function("ssd", risk_aversion=self.risk_aversion),
            "hypervolume": lambda: self._hypervolume_aggregator,
        }
        builder = builders.get(name)
        if builder is not None:
            return builder()
        return get_aggregation_function(name)

    def _use_default_aggregators(self) -> None:
        """Set default aggregation configuration."""
        self.logger.info("Using default aggregation configuration")

        # Default configuration: median for per-family (robust to outliers),
        # SSD for composite (penalises imbalance across families).
        self.agg_layers_1_2_name = "median"
        self.agg_layers_3_4_name = "ssd"

        self.fidelity_aggs = {
            "level_1": "median",
            "level_2": "median",
            "level_3": "ssd",
        }
        self.privacy_aggs = {
            "level_1": "median",
            "level_2": "median",
            "level_3": "ssd",
        }
        self.utility_aggs = {
            "level_1": "median",
            "level_2": "median",
            "level_3": "ssd",
        }
        self.composite_agg_name = "ssd"

    def _apply_aggregation(self, values: list[float], layer: str) -> float:
        """
        Apply the appropriate aggregation function based on layer.

        Args:
            values: list of values to aggregate
            layer: "1_2" or "3_4" to indicate which aggregation to use

        Returns:
            Aggregated value
        """
        valid_values = self._filter_finite_values(values)
        if not valid_values:
            return float("nan")

        if layer == "1_2":
            return float(self.agg_func_1_2(valid_values))
        if layer == "3_4":
            return float(self.agg_func_3_4(valid_values))
        return statistics.mean(valid_values)

    def _apply_composite_aggregation(self, values: list[float]) -> float:
        """Apply the composite aggregation function (families → final score).

        Uses ``agg_func_composite`` which may differ from ``agg_func_3_4``
        when auto-tuning selects a different composite aggregator.
        """
        valid_values = self._filter_finite_values(values)
        if not valid_values:
            return float("nan")
        return float(self.agg_func_composite(valid_values))

    @staticmethod
    def _hypervolume_aggregator(values: list[float]) -> float:
        """
        Hypervolume indicator (simplified implementation).

        Args:
            values: list of metric values in [0, 1]

        Returns:
            Hypervolume indicator (geometric mean as approximation)
        """
        values_array = np.array(values)
        values_array = np.clip(values_array, 1e-10, 1.0)  # Avoid zeros

        if len(values_array) == 0:
            return 0.0

        return float(stats.gmean(values_array))

    def _apply_calibration_normalization(self, aggregates: dict[str, Any]) -> dict[str, Any]:
        """
        Apply calibration normalization to family scores.

        Args:
            aggregates: Aggregates dictionary with raw family scores

        Returns:
            Updated aggregates with calibrated scores
        """
        calibrated_aggregates = aggregates.copy()
        calibrated_family_scores = {}

        for family in FAMILIES:
            score_key = f"{family}_score"
            if score_key in aggregates:
                raw_score = aggregates[score_key]
                if not np.isfinite(raw_score):
                    continue

                try:
                    calibrated_score = self.calibration_bounds.normalize_with_bounds(
                        family, raw_score
                    )

                    calibrated_aggregates[score_key] = calibrated_score
                    calibrated_aggregates[f"{family}_score_raw"] = raw_score
                    calibrated_family_scores[family] = calibrated_score
                except Exception as e:
                    self.logger.warning("Calibration normalization failed for %s: %s", family, e)
                    calibrated_aggregates[score_key] = raw_score
                    calibrated_family_scores[family] = raw_score

        # Recalculate composite score with calibrated values
        if calibrated_family_scores:
            family_scores_array = list(calibrated_family_scores.values())
            composite = self._apply_composite_aggregation(family_scores_array)
            calibrated_aggregates["composite_score"] = float(composite)
            calibrated_aggregates["composite_score_calibrated"] = True

        return calibrated_aggregates
