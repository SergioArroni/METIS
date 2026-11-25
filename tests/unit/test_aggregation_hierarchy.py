"""Hierarchical aggregation (F2), aggregation weights (A3) and failure flags (F9)."""

import math

import numpy as np
import pytest

from metis.application.aggregator import Aggregator
from metis.application.pipeline.aggregator import ResultAggregator
from metis.domain.entities import EvalPlan, MetricResult
from metis.domain.taxonomy import FIDELITY_METRICS
from metis.infrastructure.metrics.aggregation.stochastic_dominance import (
    DEFAULT_RISK_AVERSION,
    ssd_score,
)

GOOD = 0.9
BAD = 0.3
# λ = 5 over 32 metrics in three categories: one SSD over the raw metrics (flat) with a
# category bad, and one over the three category scores (hierarchical).
MARGINAL_FLAT, GLOBAL_FLAT, HIER = 0.4075, 0.6948, 0.5007


def _fidelity_ids_by_category() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for category, data in FIDELITY_METRICS.items():
        ids = list(data.get("metrics", []))
        for sub_ids in data.get("subcategories", {}).values():
            ids.extend(sub_ids)
        out[category] = ids
    return out


def _results(bad_category: str, family: str = "fidelity") -> list[MetricResult]:
    return [
        MetricResult(
            id=mid,
            value=BAD if category == bad_category else GOOD,
            details={},
            family=family,
            purpose_tags=set(),
        )
        for category, ids in _fidelity_ids_by_category().items()
        for mid in ids
    ]


def _aggregate(results, weights=None, **kwargs):
    return Aggregator(**kwargs).aggregate(
        results, EvalPlan(metric_ids=[r.id for r in results]), weights
    )


def test_profile_has_18_marginal_11_conditional_3_global_metrics() -> None:
    # global: distances (mmd, energy_distance) and detection (discriminator); correlation_matrix
    # in num_num and outliers_coverage in tails (§25-§26, 2026-11-03)
    sizes = {c: len(ids) for c, ids in _fidelity_ids_by_category().items()}
    assert sizes == {"marginal": 18, "conditional": 11, "global": 3}


@pytest.mark.parametrize(
    ("bad_category", "flat_expected", "hier_expected"),
    [
        # Audit section 3.3, F2: one category at 0.3, the rest at 0.9 (the audit's
        # 0.412 / 0.652 were over 31 metrics and three categories; now 32, with the
        # discriminator in global).
        ("marginal", MARGINAL_FLAT, HIER),
        ("global", GLOBAL_FLAT, HIER),
        ("conditional", None, HIER),
    ],
)
def test_family_score_is_hierarchical_not_flat(bad_category, flat_expected, hier_expected) -> None:
    # The audit's numbers were computed with λ = 5 (METIS's default until 2026-10-28, §22);
    # the point, hierarchical ≠ flat, does not depend on λ.
    lam = 5.0
    results = _results(bad_category)
    flat = ssd_score(np.array([r.value for r in results]), risk_aversion=lam)
    if flat_expected is not None:
        # What the flat aggregation (one SSD over the raw metrics) gives.
        assert flat == pytest.approx(flat_expected, abs=1e-3)

    aggregates = _aggregate(results, risk_aversion=lam)

    # Hierarchical: SSD over the three category scores [0.9, 0.9, 0.3].
    assert aggregates["fidelity_score"] == pytest.approx(hier_expected, abs=1e-3)
    assert aggregates["fidelity_score"] == pytest.approx(
        ssd_score(np.array([GOOD, GOOD, BAD]), risk_aversion=lam)
    )
    # A bad category weighs the same whatever its number of metrics.
    assert aggregates["hierarchy"]["fidelity"]["score"] == aggregates["fidelity_score"]
    cats = aggregates["hierarchy"]["fidelity"]["categories"]
    assert cats[bad_category]["score"] == pytest.approx(BAD)


def test_category_score_aggregates_subcategory_scores() -> None:
    # tails all 0.2, scales and coverage 1.0 -> category median of [0.2, 1, 1] = 1.0
    # (flat median over the 18 metrics would also be 1.0, so use mean to tell them apart)
    results = []
    for sub, ids in FIDELITY_METRICS["marginal"]["subcategories"].items():
        for mid in ids:
            v = 0.2 if sub == "tails" else 1.0
            results.append(MetricResult(id=mid, value=v, details={}, family="fidelity"))
    agg = Aggregator(
        optimal_aggregators_path=None,
    )
    agg.fidelity_aggs = {"level_1": "mean", "level_2": "mean", "level_3": "mean"}
    out = agg.aggregate(results, EvalPlan(metric_ids=[r.id for r in results]))
    # Hierarchical mean of subcategory means: (0.2 + 1 + 1) / 3; flat mean: (7*0.2 + 11) / 18
    assert out["hierarchy"]["fidelity"]["categories"]["marginal"]["score"] == pytest.approx(
        (0.2 + 1 + 1) / 3
    )
    assert out["fidelity_score"] == pytest.approx((0.2 + 1 + 1) / 3)


def _two_families() -> list[MetricResult]:
    return [
        MetricResult(id="fidelity.ks", value=0.9, details={}, family="fidelity"),
        MetricResult(id="utility.tstr", value=0.2, details={}, family="utility"),
        MetricResult(id="privacy.dcr", value=0.8, details={}, family="privacy"),
    ]


def test_weights_change_the_composite() -> None:
    results = _two_families()
    base = _aggregate(results)
    no_utility = _aggregate(results, weights={"utility": 0})

    assert no_utility["composite_score"] != pytest.approx(base["composite_score"])
    assert no_utility["composite_score"] == pytest.approx(
        ssd_score(np.array([0.9, 0.8]), risk_aversion=DEFAULT_RISK_AVERSION)
    )
    assert "utility" not in no_utility["composite_families"]
    # The family score itself is still reported.
    assert no_utility["utility_score"] == pytest.approx(0.2)


def test_unequal_weights_use_weighted_ssd() -> None:
    results = _two_families()
    heavy_fidelity = _aggregate(results, weights={"fidelity": 3, "utility": 1, "privacy": 1})
    base = _aggregate(results)
    assert heavy_fidelity["composite_score"] > base["composite_score"]
    assert heavy_fidelity["composite_weights"] == {"fidelity": 3.0, "privacy": 1.0, "utility": 1.0}


def test_equal_weights_leave_the_composite_unchanged() -> None:
    results = _two_families()
    base = _aggregate(results)
    same = _aggregate(results, weights={"fidelity": 2, "utility": 2, "privacy": 2})
    assert same["composite_score"] == pytest.approx(base["composite_score"])


def test_category_weights_apply_inside_the_family() -> None:
    results = _results("global")
    out = _aggregate(results, weights={"fidelity.global": 0})
    assert out["fidelity_score"] == pytest.approx(GOOD)


@pytest.mark.parametrize(
    "weights",
    [{"utilty": 0}, {"fidelity.globl": 1}, {"fidelity": -1}, {"fidelity": "a lot"}],
)
def test_invalid_weights_raise_a_clear_error(weights) -> None:
    with pytest.raises(ValueError, match="aggregation.weights|Unknown aggregation weight"):
        _aggregate(_two_families(), weights=weights)


def test_all_zero_weights_raise() -> None:
    with pytest.raises(ValueError, match="weight 0 to every family"):
        _aggregate(_two_families(), weights={"fidelity": 0, "utility": 0, "privacy": 0})


def test_pipeline_aggregator_passes_config_weights() -> None:
    results = _two_families()
    plan = EvalPlan(metric_ids=[r.id for r in results])
    base = ResultAggregator().aggregate(results, plan, {})
    weighted = ResultAggregator().aggregate(
        results, plan, {"aggregation": {"weights": {"utility": 0}}}
    )
    assert weighted["composite_score"] != pytest.approx(base["composite_score"])


def test_failed_global_metric_marks_the_family_incomplete() -> None:
    results = _results("none")
    results[-1] = MetricResult(
        id=results[-1].id,
        value=float("nan"),
        details={"error": "boom"},
        family="fidelity",
    )
    out = _aggregate(results)
    assert out["fidelity_complete"] is False
    assert out["fidelity_failed_metrics"] == [results[-1].id]
    assert out["hierarchy"]["fidelity"]["complete"] is False
    assert math.isfinite(out["fidelity_score"])


class _BrokenBounds:
    """Bounds for every metric; normalising privacy.dcr raises."""

    optimal_aggregators = None

    def __contains__(self, metric_id):
        return True

    def is_calibratable(self, metric_id):
        return True

    def normalize_with_bounds(self, metric_id, score):
        if metric_id == "privacy.dcr":
            raise RuntimeError("no bounds for privacy.dcr")
        return score


def test_calibration_failure_is_recorded_not_swallowed() -> None:
    results = _two_families()
    out = Aggregator(calibration_bounds=_BrokenBounds()).aggregate(
        results, EvalPlan(metric_ids=[r.id for r in results])
    )
    assert "privacy.dcr" in out["calibration_errors"]
    assert out["composite_score_calibrated"] is False
    assert out["privacy_score"] == pytest.approx(0.8)  # raw value kept


# ------------------------------------------- weights on any node of the hierarchy

HEFESTO_LEAVES = [
    "fidelity.global",
    "fidelity.marginal.tails",
    "fidelity.marginal.scales",
    "fidelity.marginal.coverage",
    "fidelity.conditional.num_num",
    "fidelity.conditional.num_cat",
    "fidelity.conditional.cat_cat",
    "utility.ml_efficiency",
    "privacy.dataset_based.attacks",
    "privacy.dataset_based.syntactic",
    "privacy.dataset_based.empirical_similarity",
    "privacy.mechanism_based",
]


def _marginal(tails: float = 0.2, other: float = 1.0) -> list[MetricResult]:
    return [
        MetricResult(
            id=mid, value=tails if sub == "tails" else other, details={}, family="fidelity"
        )
        for sub, ids in FIDELITY_METRICS["marginal"]["subcategories"].items()
        for mid in ids
    ]


def test_subcategory_weight_changes_the_category_score() -> None:
    results = _marginal()
    base = Aggregator().aggregate(results, None)
    heavy = Aggregator().aggregate(results, None, weights={"fidelity.marginal.tails": 3})
    marginal = heavy["hierarchy"]["fidelity"]["categories"]["marginal"]
    # Fidelity's level 2 is the mean: [0.2, 1, 1] -> 0.733; weighted [3, 1, 1] -> 0.52.
    assert base["hierarchy"]["fidelity"]["categories"]["marginal"]["score"] == pytest.approx(
        (0.2 + 1 + 1) / 3
    )
    assert marginal["score"] == pytest.approx((3 * 0.2 + 1 + 1) / 5)
    assert marginal["subcategories"]["tails"]["weight"] == 3.0
    assert heavy["fidelity_score"] < base["fidelity_score"]


def test_fidelity_defaults_to_means_below_the_family_so_one_subcategory_counts() -> None:
    """2026-10-20: with medians a defect seen by one of three subcategories (or by
    some of the six coverage metrics) left the category untouched."""
    agg = Aggregator()
    assert agg.fidelity_aggs == {"level_1": "mean", "level_2": "mean", "level_3": "ssd"}
    assert agg.privacy_aggs == {"level_1": "mean", "level_2": "mean", "level_3": "ssd"}  # §20
    ok = agg.aggregate(_marginal(tails=1.0), None)["hierarchy"]["fidelity"]["categories"]
    bad = agg.aggregate(_marginal(tails=0.2), None)["hierarchy"]["fidelity"]["categories"]
    assert ok["marginal"]["score"] == pytest.approx(1.0)
    assert bad["marginal"]["score"] == pytest.approx((0.2 + 1 + 1) / 3)


def test_subcategory_weight_with_a_mean_aggregator_is_a_weighted_mean() -> None:
    agg = Aggregator()
    agg.fidelity_aggs = {"level_1": "mean", "level_2": "mean", "level_3": "mean"}
    out = agg.aggregate(_marginal(), None, weights={"fidelity.marginal.tails": 3})
    expected = (3 * 0.2 + 1.0 + 1.0) / 5
    assert out["hierarchy"]["fidelity"]["categories"]["marginal"]["score"] == pytest.approx(
        expected
    )
    zero = agg.aggregate(_marginal(), None, weights={"fidelity.marginal.tails": 0})
    assert zero["fidelity_score"] == pytest.approx(1.0)


def test_metric_weight_by_id_or_full_path() -> None:
    agg = Aggregator()
    agg.fidelity_aggs = {"level_1": "mean", "level_2": "mean", "level_3": "mean"}
    results = [
        MetricResult(id="ks", value=0.0, details={}, family="fidelity"),
        MetricResult(id="wasserstein", value=1.0, details={}, family="fidelity"),
    ]
    by_id = agg.aggregate(results, None, weights={"fidelity.ks": 0})
    by_path = agg.aggregate(results, None, weights={"fidelity.marginal.tails.ks": 0})
    assert by_id["fidelity_score"] == pytest.approx(1.0)
    assert by_path["fidelity_score"] == pytest.approx(1.0)
    tails = by_id["hierarchy"]["fidelity"]["categories"]["marginal"]["subcategories"]["tails"]
    assert tails["metric_weights"] == {"fidelity.ks": 0.0}
    with pytest.raises(ValueError, match="same node twice"):
        agg.aggregate(results, None, weights={"fidelity.ks": 1, "fidelity.marginal.tails.ks": 2})


def test_weight_of_a_metric_by_its_full_path_in_global() -> None:
    agg = Aggregator()
    agg.fidelity_aggs = {"level_1": "mean", "level_2": "mean", "level_3": "mean"}
    results = [
        MetricResult(id="fidelity.mmd", value=0.0, details={}, family="fidelity"),
        MetricResult(id="fidelity.energy_distance", value=1.0, details={}, family="fidelity"),
    ]
    out = agg.aggregate(results, None, weights={"fidelity.global.distances.mmd": 3})
    assert out["fidelity_score"] == pytest.approx(0.25)
    distances = out["hierarchy"]["fidelity"]["categories"]["global"]["subcategories"]["distances"]
    assert distances["metric_weights"] == {"fidelity.mmd": 3.0}


def test_every_hefesto_leaf_is_a_valid_weight_key() -> None:
    _, nodes = Aggregator._parse_weights(dict.fromkeys(HEFESTO_LEAVES, 2.0))
    assert len(nodes) == len(HEFESTO_LEAVES)
    # ``utility.ml_efficiency`` is the category, not the metric of the same name.
    assert nodes[("utility", "ml_efficiency")] == 2.0


def test_unknown_node_lists_the_valid_keys() -> None:
    with pytest.raises(ValueError, match=r"fidelity\.marginal\.tails") as err:
        Aggregator().aggregate(_marginal(), None, weights={"fidelity.marginal.tail": 2})
    assert "fidelity.marginal.tail'" in str(err.value)


def test_a_category_whose_subcategories_are_all_excluded_has_no_score() -> None:
    weights = {f"fidelity.marginal.{s}": 0 for s in ("tails", "scales", "coverage")}
    out = Aggregator().aggregate(_marginal(), None, weights=weights)
    assert math.isnan(out["hierarchy"]["fidelity"]["categories"]["marginal"]["score"])


def test_pipeline_aggregator_accepts_subcategory_weights() -> None:
    results = _marginal()
    plan = EvalPlan(metric_ids=[r.id for r in results])
    base = ResultAggregator().aggregate(results, plan, {})
    weighted = ResultAggregator().aggregate(
        results, plan, {"aggregation": {"weights": {"fidelity.marginal.tails": 3}}}
    )
    assert weighted["fidelity_score"] < base["fidelity_score"]


def test_default_risk_aversion_is_three_and_a_total_failure_does_not_pass() -> None:
    """§22: with λ = 3 a table perfect in two families and null in the third (1, 1, 0) scores
    below one that is mediocre in all three (0.5), and it is not just the worst family."""
    assert DEFAULT_RISK_AVERSION == 3.0
    assert Aggregator().risk_aversion == 3.0
    score = ssd_score(np.array([1.0, 1.0, 0.0]))
    assert 0.0 < score < 0.5
