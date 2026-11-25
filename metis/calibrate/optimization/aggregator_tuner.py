"""Aggregator hyperparameter tuning.

Tests different aggregation methods to find which best approximates
theoretical bounds (real_vs_real -> 1.0, real_vs_noise -> 0.0).

Two entry points:
- ``tune_from_metrics``: receives per-metric raw values per iteration and
  re-aggregates with each candidate function to find the best per-family
  aggregator **and** the best composite aggregator.  This is the
  recommended path because it ensures bounds are computed with the same
  aggregation that will be used at evaluation time.

Family scores are computed by :class:`metis.application.aggregator.Aggregator`
itself (metrics -> subcategory -> category -> family, with the candidate
aggregator at every level and ``aggregation.weights`` / ``risk_aversion``
from the config).  The tuner used to apply each candidate to the *flat*
vector of metric values while the evaluation is hierarchical, so the bounds
were computed on a different statistic than the one they normalise:
``s_cal = (s_hier - l_flat) / (u_flat - l_flat)``.
- ``tune``: receives pre-aggregated family scores.  Kept for backward
  compatibility but only useful when per-metric data is unavailable.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from metis.infrastructure.metrics.aggregation.stochastic_dominance import DEFAULT_RISK_AVERSION
from metis.shared.aggregation_registry import AGGREGATION_FUNCTIONS

# Families whose reference orientation is known: real-vs-real must score high
# and real-vs-noise low.  Only for the others (privacy) is an aggregator that
# reverses the references accepted (``min(normal, inverted)`` error); for
# fidelity / utility that reversal used to score zero error and win.
_KNOWN_ORIENTATION_FAMILIES = frozenset({"fidelity", "utility"})


class HierarchicalScorer:
    """Family and composite scores exactly as the evaluation computes them.

    Wraps :class:`metis.application.aggregator.Aggregator` through its public
    constructor (``optimal_aggregators_path``) and ``aggregate`` so the tuner
    and the calibration bounds use the same hierarchy, weights and risk
    aversion as ``Aggregator.aggregate``.
    """

    def __init__(
        self,
        risk_aversion: float = DEFAULT_RISK_AVERSION,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.risk_aversion = float(risk_aversion)
        self.weights = dict(weights) if weights else None
        self._aggregators: dict[tuple, Any] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> HierarchicalScorer:
        agg_cfg = (config or {}).get("aggregation") or {}
        return cls(agg_cfg.get("risk_aversion", DEFAULT_RISK_AVERSION), agg_cfg.get("weights"))

    def _aggregator(self, optimal: dict[str, str]):
        key = tuple(sorted(optimal.items()))
        agg = self._aggregators.get(key)
        if agg is None:
            from metis.application.aggregator import Aggregator

            with tempfile.TemporaryDirectory(prefix="metis_tuner_") as tmp:
                path = Path(tmp) / "aggregators.json"
                path.write_text(json.dumps({"optimal": optimal}), encoding="utf-8")
                agg = Aggregator(
                    risk_aversion=self.risk_aversion, optimal_aggregators_path=str(path)
                )
            self._aggregators[key] = agg
        return agg

    def aggregate(
        self,
        values_by_family: dict[str, dict[str, float]],
        optimal: dict[str, str],
        *,
        family_weights: bool = True,
    ) -> dict[str, Any]:
        """``Aggregator.aggregate`` on one iteration's ``{family: {metric_id: value}}``."""
        from metis.domain.entities import EvalPlan, MetricResult

        results = [
            MetricResult(id=mid, value=float(v), details={}, family=family)
            for family, values in values_by_family.items()
            for mid, v in values.items()
        ]
        if not results:
            return {}
        plan = EvalPlan(metric_ids=[r.id for r in results])
        weights = self._weights_for(values_by_family, family_weights)
        return self._aggregator(optimal).aggregate(results, plan, weights)

    def _weights_for(
        self, values_by_family: dict[str, dict[str, float]], family_weights: bool
    ) -> dict | None:
        """Configured weights for the families present.

        Category weights (``family.category``) shape the family score; family
        weights only the composite, so they are left out when scoring a
        single family (a family weighted 0 would make the composite fail).
        """
        if not self.weights:
            return None
        present = set(values_by_family)
        kept = {}
        for key, w in self.weights.items():
            family, _, category = str(key).partition(".")
            if family in present and (category or family_weights):
                kept[key] = w
        return kept or None

    def family_score(self, family: str, values: dict[str, float], agg_name: str) -> float:
        aggregates = self.aggregate(
            {family: values}, {family: agg_name, "composite": "mean"}, family_weights=False
        )
        return float(aggregates.get(f"{family}_score", float("nan")))


class AggregatorTuner:
    """
    Tune aggregation functions for optimal calibration.

    Tests various aggregation methods and finds the one that best
    represents theoretical bounds for each family.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        scorer: HierarchicalScorer | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.aggregation_functions = AGGREGATION_FUNCTIONS
        self.scorer = scorer or HierarchicalScorer()
        self.results: dict[str, Any] = {}

        # Early stopping tolerance
        self.early_stop_tol = 1e-6

        # Available aggregators for per-family optimisation.  ``fsd`` is not a
        # candidate: ``fsd_score_for_row`` is the arithmetic mean, so it tied
        # with ``mean`` on every input and the strict ``<`` never let it win.
        self.available_aggregators = [
            "mean",
            "median",
            "trimmed_mean_10",
            "ssd",
        ]

        # Composite cannot use geometric_mean (zero-product problem:
        # gm(1, 1, 0) = 0, which makes real_data composite worse than noise).
        self.composite_aggregators = [
            "mean",
            "median",
            "trimmed_mean_10",
            "ssd",
        ]

    @staticmethod
    def is_degenerate_composite_aggregator(agg_func: Any) -> bool:
        """Reject composite operators that ignore a collapsed family.

        A final composite should stay strictly inside the open interval
        ``(0, 1)`` for mixed extreme reference vectors such as
        ``[1, 1, 0]`` and ``[0, 0, 1]``. If an operator maps either case
        to an endpoint, one family can hit its worst case without any
        penalty at L4, which is exactly the Airbnb failure mode.
        """
        mixed_upper = float(agg_func([1.0, 1.0, 0.0]))
        mixed_lower = float(agg_func([0.0, 0.0, 1.0]))
        tol = 1e-9

        if not np.isfinite(mixed_upper) or not np.isfinite(mixed_lower):
            return True

        return (
            mixed_upper <= tol
            or mixed_upper >= 1.0 - tol
            or mixed_lower <= tol
            or mixed_lower >= 1.0 - tol
        )

    # -----------------------------------------------------------------
    # Primary API: tune from raw per-metric values
    # -----------------------------------------------------------------

    def tune_from_metrics(
        self,
        upper_metric_data: dict[str, list[dict[str, float]]],
        lower_metric_data: dict[str, list[dict[str, float]]],
    ) -> dict[str, Any]:
        """Find the optimal per-family aggregator **and** composite aggregator.

        This method takes the raw, per-metric values from each calibration
        iteration and tries every candidate aggregation function to combine
        them into a family score.  The aggregator whose family scores are
        closest to the theoretical limits (1.0 for the upper strategy,
        0.0 for the lower strategy -- or inverted for privacy) wins.

        Args:
            upper_metric_data: ``{family: [[metric_values_iter_1], ...]}``.
                Raw per-metric values from the **upper** bound strategy
                (Real-vs-Real) for every iteration.
            lower_metric_data: Same structure for the **lower** bound
                strategy (Real-vs-Noise).

        Returns:
            Dictionary with keys ``optimal``, ``target_upper``,
            ``target_lower``, ``available_aggregators``.
            ``optimal`` maps each family to the best aggregator name and
            contains a ``composite`` key.
        """
        self.logger.info("=" * 70)
        self.logger.info("AGGREGATOR TUNING (per-metric re-aggregation)")
        self.logger.info("=" * 70)
        self.logger.info("Families: %s", sorted(upper_metric_data.keys()))
        self.logger.info("Candidates: %s", self.available_aggregators)

        optimal_config: dict[str, Any] = {}

        # ---- Per-family optimisation ----
        for family in sorted(upper_metric_data.keys()):
            upper_iters = upper_metric_data[family]
            lower_iters = lower_metric_data.get(family, [])

            if not upper_iters or not lower_iters:
                self.logger.warning("%s: missing iteration data, skipping", family)
                continue

            self.logger.info(
                "\n--- %s (%d upper, %d lower iters, %d metrics) ---",
                family.upper(),
                len(upper_iters),
                len(lower_iters),
                len(upper_iters[0]) if upper_iters else 0,
            )

            best_name, best_error = self._pick_best_family_aggregator(
                upper_iters,
                lower_iters,
                family,
            )

            optimal_config[family] = best_name
            self.logger.info("  -> Best for %s: %s (error=%.6f)", family, best_name, best_error)

        # ---- Composite optimisation ----
        self.logger.info("\n--- COMPOSITE ---")
        best_composite = self._pick_best_composite(
            upper_metric_data,
            lower_metric_data,
            optimal_config,
        )
        optimal_config["composite"] = best_composite

        # ---- Persist ----
        self.results = {
            "optimal": optimal_config,
            "target_upper": 1.0,
            "target_lower": 0.0,
            "available_aggregators": self.available_aggregators,
        }
        self.logger.info("\n" + "=" * 70)
        self.logger.info("TUNING COMPLETE: %s", optimal_config)
        self.logger.info("=" * 70)
        return self.results

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _pick_best_family_aggregator(
        self,
        upper_iters: list[dict[str, float]],
        lower_iters: list[dict[str, float]],
        family: str,
    ) -> tuple[str, float]:
        """Test every candidate aggregator on per-metric values.

        For each candidate ``f``:
          upper_scores = [f(iter_values) for iter_values in upper_iters]
          lower_scores = [f(iter_values) for iter_values in lower_iters]

        Each ``iter_values`` is a dict ``{metric_id: value}``.  Only the
        numeric values are passed to the aggregation function.

        We detect the orientation automatically:
          * Normal (fidelity/utility): upper_mean ~ 1.0, lower_mean ~ 0.0
          * Inverted (privacy):        upper_mean ~ 0.0, lower_mean ~ 1.0

        The error is (upper_mean - target_high)^2 + (lower_mean - target_low)^2.
        """
        best_name = "median"
        best_error = float("inf")

        for agg_name in self.available_aggregators:
            try:
                upper_scores = [
                    self.scorer.family_score(family, vals, agg_name) for vals in upper_iters
                ]
                lower_scores = [
                    self.scorer.family_score(family, vals, agg_name) for vals in lower_iters
                ]
            except Exception as exc:  # noqa: BLE001 - a candidate that fails is skipped
                self.logger.warning("    %15s: failed (%s)", agg_name, exc)
                continue

            upper_mean = float(np.nanmean(upper_scores)) if upper_scores else float("nan")
            lower_mean = float(np.nanmean(lower_scores)) if lower_scores else float("nan")
            if not (np.isfinite(upper_mean) and np.isfinite(lower_mean)):
                continue

            error_normal = (upper_mean - 1.0) ** 2 + (lower_mean - 0.0) ** 2
            if family in _KNOWN_ORIENTATION_FAMILIES:
                error = error_normal
            else:
                error_invert = (upper_mean - 0.0) ** 2 + (lower_mean - 1.0) ** 2
                error = min(error_normal, error_invert)

            self.logger.info(
                "    %15s: upper_mean=%.4f lower_mean=%.4f err=%.6f",
                agg_name,
                upper_mean,
                lower_mean,
                error,
            )

            if error < best_error:
                best_error = error
                best_name = agg_name

            if best_error < self.early_stop_tol:
                break

        return best_name, best_error

    def _pick_best_composite(
        self,
        upper_metric_data: dict[str, list[dict[str, float]]],
        lower_metric_data: dict[str, list[dict[str, float]]],
        family_aggs: dict[str, str],
    ) -> str:
        """Find the composite aggregator that best combines tuned family scores.

        For each composite candidate, the per-family scores are first
        computed using the already-selected per-family aggregator, then
        the composite candidate combines those family scores.
        """
        families = sorted(f for f in family_aggs if f != "composite")
        if not families:
            return "ssd"
        n_upper = min(len(upper_metric_data[f]) for f in families)
        n_lower = min(len(lower_metric_data[f]) for f in families)

        def _iteration(data, i):
            return {f: data[f][i] for f in families}

        best_name = "ssd"
        best_error = float("inf")

        candidate_names = []
        for comp_name in self.composite_aggregators:
            comp_func = self.aggregation_functions[comp_name]
            if self.is_degenerate_composite_aggregator(comp_func):
                self.logger.info(
                    "    composite %15s: skipped (degenerate on mixed extremes)",
                    comp_name,
                )
                continue
            candidate_names.append(comp_name)

        if not candidate_names:
            self.logger.warning(
                "No non-degenerate composite aggregators available; falling back to full candidate set"
            )
            candidate_names = list(self.composite_aggregators)

        for comp_name in candidate_names:
            optimal = {**{f: family_aggs[f] for f in families}, "composite": comp_name}
            try:
                upper_composites = [
                    float(
                        self.scorer.aggregate(_iteration(upper_metric_data, i), optimal).get(
                            "composite_score", float("nan")
                        )
                    )
                    for i in range(n_upper)
                ]
                lower_composites = [
                    float(
                        self.scorer.aggregate(_iteration(lower_metric_data, i), optimal).get(
                            "composite_score", float("nan")
                        )
                    )
                    for i in range(n_lower)
                ]
            except Exception as exc:  # noqa: BLE001 - a candidate that fails is skipped
                self.logger.warning("    composite %15s: failed (%s)", comp_name, exc)
                continue

            upper_mean = float(np.nanmean(upper_composites))
            lower_mean = float(np.nanmean(lower_composites))
            if not (np.isfinite(upper_mean) and np.isfinite(lower_mean)):
                continue

            error_normal = (upper_mean - 1.0) ** 2 + (lower_mean - 0.0) ** 2
            error_invert = (upper_mean - 0.0) ** 2 + (lower_mean - 1.0) ** 2
            error = min(error_normal, error_invert)

            self.logger.info(
                "    composite %15s: upper=%.4f lower=%.4f err=%.6f",
                comp_name,
                upper_mean,
                lower_mean,
                error,
            )

            if error < best_error:
                best_error = error
                best_name = comp_name

            if best_error < self.early_stop_tol:
                break

        self.logger.info("  -> Best composite: %s (error=%.6f)", best_name, best_error)
        return best_name

    # -----------------------------------------------------------------
    # Re-aggregation helper (used by MetricCalibrator)
    # -----------------------------------------------------------------

    @staticmethod
    def reaggregate(
        metric_data: dict[str, list[dict[str, float]]],
        family_aggs: dict[str, str],
        scorer: HierarchicalScorer | None = None,
    ) -> dict[str, list[float]]:
        """Re-aggregate per-metric data with specific aggregators.

        Family scores come from the evaluation's hierarchical ``Aggregator``
        (see :class:`HierarchicalScorer`), so the calibration bounds are on
        the same scale as the scores they normalise.

        Args:
            metric_data: ``{family: [{metric_id: value, ...}, ...]}``
            family_aggs: ``{family: aggregator_name}``
            scorer: Weights / risk aversion of the evaluation (default: none / 5).

        Returns:
            ``{family: [score_iter1, score_iter2, ...]}``
        """
        scorer = scorer or HierarchicalScorer()
        results: dict[str, list[float]] = {}
        for family, iterations in metric_data.items():
            agg_name = family_aggs.get(family, "median")
            results[family] = [scorer.family_score(family, vals, agg_name) for vals in iterations]
        return results

    # -----------------------------------------------------------------
    # Legacy API (pre-aggregated family scores)
    # -----------------------------------------------------------------

    def tune(
        self,
        upper_bound_results: dict[str, list[float]],
        lower_bound_results: dict[str, list[float]],
        target_upper: float = 1.0,
        target_lower: float = 0.0,
    ) -> dict[str, Any]:
        """Find optimal aggregation function from pre-aggregated scores.

        .. deprecated:: Use :meth:`tune_from_metrics` instead.
        """
        self.logger.info("=" * 70)
        self.logger.info("AGGREGATOR TUNING (legacy -- pre-aggregated scores)")
        self.logger.info("=" * 70)

        optimal_config: dict[str, Any] = {}

        for family in sorted(upper_bound_results.keys()):
            best = self._optimize_single_level(
                upper_bound_results[family],
                lower_bound_results[family],
                target_upper,
                target_lower,
                level_name=family,
            )
            optimal_config[family] = best

        upper_family_scores = [np.mean(v) for v in upper_bound_results.values()]
        lower_family_scores = [np.mean(v) for v in lower_bound_results.values()]
        composite_optimal = self._optimize_single_level(
            upper_family_scores,
            lower_family_scores,
            target_upper,
            target_lower,
            level_name="Composite",
            candidate_aggregators=self.composite_aggregators,
        )
        optimal_config["composite"] = composite_optimal

        self.results = {
            "optimal": optimal_config,
            "target_upper": target_upper,
            "target_lower": target_lower,
            "available_aggregators": self.available_aggregators,
        }
        return self.results

    def _optimize_single_level(
        self,
        upper_values: list[float],
        lower_values: list[float],
        target_upper: float,
        target_lower: float,
        level_name: str = "Level",
        candidate_aggregators: list[str] | None = None,
    ) -> str:
        """Find the best aggregator for pre-aggregated values (legacy)."""
        best_aggregator = "median"
        best_error = float("inf")

        candidates = candidate_aggregators or self.available_aggregators
        for agg_name in candidates:
            agg_func = self.aggregation_functions[agg_name]

            try:
                upper_agg = agg_func(upper_values)
                upper_error = (upper_agg - target_upper) ** 2
            except Exception:
                upper_error = float("inf")

            try:
                lower_agg = agg_func(lower_values)
                lower_error = (lower_agg - target_lower) ** 2
            except Exception:
                lower_error = float("inf")

            total_error = upper_error + lower_error

            if total_error < best_error:
                best_error = total_error
                best_aggregator = agg_name

            if best_error < self.early_stop_tol:
                return best_aggregator

        return best_aggregator

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save tuning results to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        self.logger.info("Results saved to: %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> dict[str, Any]:
        """Load optimal aggregators from JSON."""
        path = Path(filepath)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        return data.get("optimal", {})
