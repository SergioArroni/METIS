"""Result aggregation and scoring logic."""

import statistics
from typing import Any

import numpy as np
from scipy import stats

from ..domain.contracts import BoundsStorage
from ..domain.entities import EvalPlan, MetricResult
from ..domain.taxonomy import (
    FAMILIES,
    FIDELITY_METRICS,
    PRIVACY_METRICS,
    UTILITY_METRICS,
    get_metric_hierarchy,
)
from ..infrastructure.metrics.aggregation.stochastic_dominance import (
    DEFAULT_RISK_AVERSION,
    weighted_fsd_score,
    weighted_ssd_score,
)
from ..infrastructure.runtime.logging import get_logger
from ..shared.aggregation_registry import get_aggregation_function
from ..shared.anchors import WARNING_Z

_TAXONOMY_BY_FAMILY: dict[str, dict] = {
    "fidelity": FIDELITY_METRICS,
    "privacy": PRIVACY_METRICS,
    "utility": UTILITY_METRICS,
}

# Pseudo-subcategory of the metrics listed directly under a category
# (``fidelity.global``, ``privacy.mechanism_based``).
_DIRECT = "_direct"

# A node of the hierarchy: (family,), (family, category),
# (family, category, subcategory) or (family, category, subcategory, metric_id),
# with subcategory ``_DIRECT`` for metrics that hang from the category.
NodePath = tuple[str, ...]


def _build_weight_keys() -> tuple[dict[str, NodePath], list[str]]:
    """Every key ``aggregation.weights`` accepts, mapped to its node.

    Canonical keys are the dotted paths of the taxonomy: ``fidelity``,
    ``fidelity.marginal``, ``fidelity.marginal.tails`` and, for a metric, its
    full path ``fidelity.marginal.tails.ks`` (``fidelity.global.mmd`` for a
    metric directly under a category). A metric can also be named by its id
    (``fidelity.ks``) unless the id is also a category: ``utility.ml_efficiency``
    is the category, the metric is ``utility.ml_efficiency.aggregate.ml_efficiency``.

    Returns the key map and the sorted canonical keys of families, categories
    and subcategories (what error messages list).
    """
    keys: dict[str, NodePath] = {}
    metric_ids: dict[str, NodePath] = {}
    hierarchy_keys: list[str] = []
    for family in sorted(_TAXONOMY_BY_FAMILY):
        keys[family] = (family,)
        hierarchy_keys.append(family)
        for category, data in _TAXONOMY_BY_FAMILY[family].items():
            cat_key = f"{family}.{category}"
            keys[cat_key] = (family, category)
            hierarchy_keys.append(cat_key)
            groups = {_DIRECT: data.get("metrics", []), **data.get("subcategories", {})}
            for sub, ids in groups.items():
                if sub != _DIRECT:
                    keys[f"{cat_key}.{sub}"] = (family, category, sub)
                    hierarchy_keys.append(f"{cat_key}.{sub}")
                prefix = cat_key if sub == _DIRECT else f"{cat_key}.{sub}"
                for metric_id in ids:
                    path = (family, category, sub, metric_id)
                    keys[f"{prefix}.{metric_id.split('.', 1)[1]}"] = path
                    metric_ids[metric_id] = path
    for metric_id, path in metric_ids.items():
        keys.setdefault(metric_id, path)  # a category of the same name wins
    return keys, sorted(hierarchy_keys)


_WEIGHT_KEYS, _HIERARCHY_WEIGHT_KEYS = _build_weight_keys()


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
        risk_aversion: float = DEFAULT_RISK_AVERSION,
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
        plan: EvalPlan | None = None,  # noqa: ARG002
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate metric results into family scores and hierarchical breakdowns.

        Aggregation is hierarchical: metrics -> subcategory (``level_1``),
        subcategories -> category (``level_2``) and categories -> family
        (``level_3``); each level aggregates the *scores of the level below*, so
        a category weighs the same whatever its number of metrics. The family
        score is the top of that hierarchy; the composite aggregates families.

        Args:
            results: list of computed metric results
            plan: Evaluation plan used for the run (not needed; kept for the
                pipeline signature, so ``aggregate(results, None, weights)`` works)
            weights: Optional ``{node: weight}`` over any node of the taxonomy
                (``aggregation.weights`` in the config):

                * ``"utility"`` — a family, weighed in the composite;
                * ``"fidelity.global"`` — a category, weighed in its family;
                * ``"fidelity.marginal.tails"`` — a subcategory, in its category;
                * ``"fidelity.ks"`` or ``"fidelity.marginal.tails.ks"`` — a
                  metric, in its subcategory (``fidelity.global.mmd`` for one
                  directly under a category).

                A node's weight is relative to its siblings only (default 1):
                ``{"fidelity.marginal.tails": 3}`` makes tails count three times
                scales or coverage inside ``marginal`` and nothing else. A weight
                of 0 excludes the node; a node whose children are all excluded
                has no score. Unequal weights need a weighted aggregator (mean,
                fsd, ssd, median). Unknown keys, a metric named twice or
                invalid values raise ``ValueError`` listing the valid keys.

        Returns:
            Dictionary containing family scores, hierarchical scores, and metadata
        """
        family_weights, node_weights = self._parse_weights(weights)
        aggregates: dict[str, Any] = {}

        # Level 0: per-metric calibration, before any aggregation.
        if self.calibration_bounds:
            raw_hierarchy = self._calculate_hierarchy_scores(results, node_weights)
            for family, data in raw_hierarchy.items():
                # Uncalibrated family scores, for traceability.
                aggregates[f"{family}_score_raw"] = float(data.get("score", float("nan")))
            results, calibration = self._calibrate_results(results)
            aggregates.update(calibration)

        # Levels 1-3: metrics -> subcategories -> categories -> families
        hierarchy_scores = self._calculate_hierarchy_scores(results, node_weights)

        family_scores_dict = {}
        for family in sorted(FAMILIES):
            family_results = [r for r in results if self._family_of(r) == family]
            if not family_results:
                continue
            family_data = hierarchy_scores.get(family, {})
            family_score = float(family_data.get("score", float("nan")))
            aggregates[f"{family}_score"] = family_score
            aggregates[f"{family}_count"] = len(family_results)
            not_applicable = [r.id for r in family_results if r.status == "not_applicable"]
            if not_applicable:
                # Did not apply to these data/config: reported, not a failure.
                aggregates[f"{family}_not_applicable"] = not_applicable
            failed = family_data.get("failed_metrics", [])
            if failed:
                # The family score only reflects the metrics that worked: flag it.
                aggregates[f"{family}_failed_metrics"] = failed
                aggregates[f"{family}_complete"] = False
            if np.isfinite(family_score):
                family_scores_dict[family] = family_score
            else:
                self.logger.warning(
                    "Family '%s' score is NaN — excluded from composite",
                    family,
                )

        # Level 4: families -> composite
        self._set_composite(aggregates, family_scores_dict, family_weights)
        aggregates["hierarchy"] = hierarchy_scores

        # Add execution metadata
        aggregates["total_metrics"] = len(results)
        aggregates["successful_metrics"] = len(
            [r for r in results if self._is_valid_metric_result(r)]
        )
        aggregates["not_applicable_metrics"] = len(
            [r for r in results if r.status == "not_applicable"]
        )
        aggregates["failed_metrics"] = (
            len(results) - aggregates["successful_metrics"] - aggregates["not_applicable_metrics"]
        )

        return aggregates

    # ----- weights --------------------------------------------------------------

    @staticmethod
    def _parse_weights(
        weights: dict[str, float] | None,
    ) -> tuple[dict[str, float], dict[NodePath, float]]:
        """Validate ``aggregation.weights`` and resolve each key to its node.

        Returns ``(family_weights, {node_path: weight})`` where the node paths
        are those of ``_build_weight_keys`` (categories, subcategories, metrics).
        """
        family_weights: dict[str, float] = {}
        node_weights: dict[NodePath, float] = {}
        if not weights:
            return family_weights, node_weights
        if not isinstance(weights, dict):
            raise ValueError(f"aggregation.weights must be a mapping, got {type(weights).__name__}")

        unknown = []
        seen: dict[NodePath, str] = {}
        for key, raw in weights.items():
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"aggregation.weights['{key}'] must be a number, got {raw!r}"
                ) from None
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"aggregation.weights['{key}'] must be finite and >= 0, got {raw!r}"
                )
            path = _WEIGHT_KEYS.get(str(key))
            if path is None:
                unknown.append(key)
                continue
            if path in seen:
                raise ValueError(
                    f"aggregation.weights names the same node twice: '{seen[path]}' and '{key}'"
                )
            seen[path] = str(key)
            if len(path) == 1:
                family_weights[path[0]] = value
            else:
                node_weights[path] = value

        if unknown:
            raise ValueError(
                f"Unknown aggregation weight key(s): {sorted(map(str, unknown))}. "
                f"Valid keys: {_HIERARCHY_WEIGHT_KEYS}, or a metric id "
                "('fidelity.ks') or its full path ('fidelity.marginal.tails.ks')."
            )
        return family_weights, node_weights

    def _weighted(self, agg_name: str, values: list[float], weights: list[float]) -> float:
        """Aggregate ``values`` with ``weights`` using the named aggregator.

        Zero weights drop their value. With equal weights the plain aggregator is
        used, so results without ``weights`` are unchanged.
        """
        pairs = [
            (float(v), float(w))
            for v, w in zip(values, weights, strict=True)
            if np.isfinite(v) and w > 0
        ]
        if not pairs:
            return float("nan")
        vals = np.array([v for v, _ in pairs])
        wts = np.array([w for _, w in pairs])
        if np.allclose(wts, wts[0]):
            return float(self._build_aggregator(agg_name)(list(vals)))
        if agg_name in ("mean", "fsd"):
            return weighted_fsd_score(vals, wts)
        if agg_name == "ssd":
            return weighted_ssd_score(vals, wts, risk_aversion=self.risk_aversion)
        if agg_name == "median":
            order = np.argsort(vals)
            cum = np.cumsum(wts[order]) / wts.sum()
            return float(vals[order][np.searchsorted(cum, 0.5)])
        raise ValueError(
            f"Aggregator '{agg_name}' does not support unequal weights; "
            "use mean, fsd, ssd or median, or drop aggregation.weights"
        )

    def _set_composite(
        self,
        aggregates: dict[str, Any],
        family_scores: dict[str, float],
        family_weights: dict[str, float],
    ) -> None:
        """Families -> composite, honouring ``aggregation.weights``.

        ``composite_score`` is always present: NaN when no family has a score.
        """
        if not family_scores:
            aggregates["composite_score"] = float("nan")
            aggregates["composite_families"] = []
            return
        families = list(family_scores)
        weights = [family_weights.get(f, 1.0) for f in families]
        included = [f for f, w in zip(families, weights, strict=True) if w > 0]
        if not included:
            raise ValueError(
                f"aggregation.weights gives weight 0 to every family with a score: {families}"
            )
        composite = self._weighted(
            self.composite_agg_name, [family_scores[f] for f in families], weights
        )
        aggregates["composite_score"] = float(composite)
        aggregates["composite_families"] = included
        if family_weights:
            aggregates["composite_weights"] = dict(zip(families, weights, strict=True))

    # ----- hierarchy ------------------------------------------------------------

    @staticmethod
    def _family_of(result: MetricResult) -> str:
        if result.family in FAMILIES:
            return result.family
        return get_metric_hierarchy(result.id)["family"]

    def _level_agg_name(self, family: str, level: int) -> str:
        aggs = getattr(self, f"{family}_aggs", None) or {}
        default = {1: self.agg_layers_1_2_name, 2: self.agg_layers_1_2_name}.get(
            level, self.agg_layers_3_4_name
        )
        return aggs.get(f"level_{level}", default)

    def _aggregate_level(
        self,
        family: str,
        level: int,
        values: list[float],
        weights: list[float] | None = None,
    ) -> float:
        """Aggregate one level; zero-weight and non-finite values are dropped first."""
        if weights is None:
            weights = [1.0] * len(values)
        pairs = [
            (float(v), float(w))
            for v, w in zip(values, weights, strict=True)
            if np.isfinite(v) and w > 0
        ]
        if not pairs:
            return float("nan")
        vals = [v for v, _ in pairs]
        wts = [w for _, w in pairs]
        name = self._level_agg_name(family, level)
        if not np.allclose(wts, wts[0]):
            return self._weighted(name, vals, wts)
        tuned = getattr(self, "_family_agg_funcs", {})
        if level == 3 and family in tuned:
            return float(tuned[family](vals))
        return float(self._build_aggregator(name)(vals))

    def _calculate_hierarchy_scores(
        self,
        results: list[MetricResult],
        node_weights: dict[NodePath, float] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate aggregated scores for each level of the hierarchy.

        Each level aggregates the scores of the level below (never the raw
        metric values): metrics -> subcategory, subcategories -> category,
        categories -> family. Metrics that failed (error or non-finite value)
        are listed under ``failed_metrics`` at every level they belong to.

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
        node_weights = node_weights or {}
        grouped: dict[str, dict[str, dict[str, list[MetricResult]]]] = {}
        for r in results:
            family = self._family_of(r)
            hierarchy = get_metric_hierarchy(r.id)
            category = hierarchy["category"] if hierarchy["family"] == family else "unknown"
            subcategory = hierarchy["subcategory"] or _DIRECT
            grouped.setdefault(family, {}).setdefault(category, {}).setdefault(
                subcategory, []
            ).append(r)

        def weight(*path: str) -> float:
            return node_weights.get(path, 1.0)

        def non_default(names: list[str], path: NodePath) -> dict[str, float]:
            return {n: weight(*path, n) for n in names if (*path, n) in node_weights}

        hierarchy_scores: dict[str, Any] = {}
        for family, categories in grouped.items():
            family_data: dict[str, Any] = {"categories": {}}
            family_failed: list[str] = []
            cat_names: list[str] = []
            cat_scores: list[float] = []
            n_valid_family = 0

            for category, subcategories in categories.items():
                category_data: dict[str, Any] = {}
                category_failed: list[str] = []
                sub_names: list[str] = []
                sub_scores: list[float] = []
                n_valid_category = 0

                for subcategory, sub_results in subcategories.items():
                    valid_results = [r for r in sub_results if self._is_valid_metric_result(r)]
                    values = [r.value for r in valid_results]
                    names = [r.id.split(".")[-1] for r in valid_results]
                    metric_weights = [
                        weight(family, category, subcategory, f"{family}.{n}") for n in names
                    ]
                    failed = [
                        r.id
                        for r in sub_results
                        if not self._is_valid_metric_result(r) and r.status != "not_applicable"
                    ]
                    if failed:
                        category_failed.extend(failed)
                        self.logger.warning(
                            "Hierarchy '%s.%s.%s': skipping non-finite/errored metric(s): %s",
                            family,
                            category,
                            subcategory,
                            failed,
                        )
                    if not values:
                        continue

                    # Level 1: metrics -> subcategory
                    sub_score = self._aggregate_level(family, 1, values, metric_weights)
                    sub_names.append(subcategory)
                    sub_scores.append(sub_score)
                    n_valid_category += len(values)
                    full_ids = [f"{family}.{n}" for n in names]
                    applied = non_default(full_ids, (family, category, subcategory))

                    if subcategory == _DIRECT:
                        category_data["metrics"] = names
                        if applied:
                            category_data["metric_weights"] = applied
                    else:
                        category_data.setdefault("subcategories", {})[subcategory] = {
                            "score": sub_score,
                            "count": len(values),
                            "metrics": names,
                            **({"metric_weights": applied} if applied else {}),
                            **(
                                {"weight": weight(family, category, subcategory)}
                                if (family, category, subcategory) in node_weights
                                else {}
                            ),
                            **({"failed_metrics": failed} if failed else {}),
                        }

                # Level 2: subcategories -> category
                if sub_scores:
                    sub_weights = [weight(family, category, s) for s in sub_names]
                    category_data["score"] = self._aggregate_level(
                        family, 2, sub_scores, sub_weights
                    )
                    category_data["count"] = n_valid_category
                    cat_names.append(category)
                    cat_scores.append(category_data["score"])
                    n_valid_family += n_valid_category
                if (family, category) in node_weights:
                    category_data["weight"] = weight(family, category)
                if category_failed:
                    category_data["failed_metrics"] = category_failed
                    family_failed.extend(category_failed)

                family_data["categories"][category] = category_data

            # Level 3: categories -> family
            if cat_scores:
                weights = [weight(family, c) for c in cat_names]
                family_data["score"] = self._aggregate_level(family, 3, cat_scores, weights)
                family_data["count"] = n_valid_family
                applied = non_default(cat_names, (family,))
                if applied:
                    family_data["category_weights"] = dict(zip(cat_names, weights, strict=True))
            if family_failed:
                family_data["failed_metrics"] = family_failed
                family_data["complete"] = False

            hierarchy_scores[family] = family_data

        return hierarchy_scores

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

        # Default configuration: SSD for each family and the composite
        # (penalises imbalance).  Below it, the mean in every family.
        self.agg_layers_1_2_name = "median"
        self.agg_layers_3_4_name = "ssd"

        # Fidelity: the mean in levels 1 (metrics -> subcategory) and 2
        # (subcategories -> category).  A median of 3 subcategories, or of the
        # 6 coverage metrics, ignores a defect that only some of them see:
        # with injected defects (2400 evaluations, 3 tables x 10 ways of
        # reading the ordinals x 8 defects) the fidelity score detected 70 %
        # of them (z >= 3) with medians and 84 % with means, and no defect the
        # median detected was lost.  SSD there detects 88 % but lets a single
        # broken metric drag the whole score.  Decision of 2026-10-20,
        # ``docs/auditoria/2026-10-09-decisiones-de-metodo.md`` §19.
        self.fidelity_aggs = {
            "level_1": "mean",
            "level_2": "mean",
            "level_3": "ssd",
        }
        # Privacy: the mean too (2026-10-21).  With the median the two nearly
        # blind subcategories (syntactic, empirical_similarity) outvoted the
        # attacks: a synthetic table with 10 % of its rows copied from the
        # training data scored 0.89.  With injected leaks in 4 tables the
        # score drops 0.10 / 0.31 / 0.37 / 0.48 for 1 / 5 / 10 / 25 % copies
        # (median: 0.01 / 0.06 / 0.11 / 0.25), with no drop when nothing
        # leaks.  §20 of the method decisions.
        self.privacy_aggs = {
            "level_1": "mean",
            "level_2": "mean",
            "level_3": "ssd",
        }
        # Utility: the mean as well (2026-11-07).  It used to keep the old
        # median default, but each utility node holds one model in practice
        # (TSTR scored with the AUC, and with the log-loss), so the two agree.
        self.utility_aggs = {
            "level_1": "mean",
            "level_2": "mean",
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

    def _calibrate_results(
        self, results: list[MetricResult]
    ) -> tuple[list[MetricResult], dict[str, Any]]:
        """Rescale every calibrated metric to its own anchors (level 0).

        ``clip((s − lower) / (upper − lower), 0, 1)`` per metric (see
        ``metis.calibrate.core.bounds``); the hierarchy then aggregates the
        calibrated values.  Utility metrics and metrics without bounds keep
        their value.  A metric whose anchors are within its noise is set
        aside as ``not_applicable`` with ``details["calibration"]`` saying
        why.  A metric whose calibration raises keeps its raw value and is
        listed in ``calibration_errors`` (``composite_score_calibrated`` is
        then False).  A calibrated metric more than ``WARNING_Z`` standard
        deviations below another real sample is listed, with that deviation,
        in ``calibration_warnings``: a warning outside the score.

        Returns the calibrated results (the caller's list is not modified)
        and the calibration entries for the aggregates.
        """
        bounds = self.calibration_bounds
        calibrated: list[MetricResult] = []
        values: dict[str, float] = {}
        excluded: dict[str, str] = {}
        errors: dict[str, str] = {}
        warnings: dict[str, float] = {}
        uncalibrated: list[str] = []
        deviation = getattr(bounds, "deviation_from_real", lambda _m, _v: None)
        for r in results:
            if r.status != "ok" or self._family_of(r) == "utility":
                calibrated.append(r)
                continue
            if r.id not in bounds:
                uncalibrated.append(r.id)
                calibrated.append(r)
                continue
            is_calibratable = getattr(bounds, "is_calibratable", lambda _m: True)
            if not is_calibratable(r.id):
                reason = bounds.exclusion_reason(r.id)
                excluded[r.id] = reason
                calibrated.append(
                    MetricResult.not_applicable(
                        r.id,
                        r.family,
                        f"calibration {reason}",
                        r.purpose_tags,
                        calibration=reason,
                        raw_value=r.value,
                    )
                )
                continue
            try:
                value = float(bounds.normalize_with_bounds(r.id, r.value))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Calibration of '%s' failed (%s: %s); keeping raw value %.4f",
                    r.id,
                    type(exc).__name__,
                    exc,
                    r.value,
                )
                errors[r.id] = f"{type(exc).__name__}: {exc}"
                calibrated.append(r)
                continue
            values[r.id] = value
            z = deviation(r.id, r.value)
            if z is not None and z >= WARNING_Z:
                warnings[r.id] = z
            calibrated.append(
                MetricResult(
                    id=r.id,
                    value=value,
                    details={**r.details, "calibration": {"raw_value": r.value}},
                    family=r.family,
                    purpose_tags=r.purpose_tags,
                )
            )

        info: dict[str, Any] = {
            "calibration_method": "per_metric",
            "calibrated_metrics": values,
            "composite_score_calibrated": not errors,
        }
        if excluded:
            info["calibration_excluded"] = excluded
        if uncalibrated:
            info["calibration_missing"] = uncalibrated
        if errors:
            info["calibration_errors"] = errors
        if warnings:
            info["calibration_warnings"] = dict(sorted(warnings.items(), key=lambda kv: -kv[1]))
        return calibrated, info
