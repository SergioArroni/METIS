from metis.calibrate.optimization.aggregator_tuner import AggregatorTuner
from metis.infrastructure.metrics.aggregation.stochastic_dominance import DEFAULT_RISK_AVERSION


def _iterations(value: float, n: int = 5) -> list[dict[str, float]]:
    return [{"metric": value} for _ in range(n)]


def test_composite_tuning_rejects_degenerate_median_on_mixed_extremes() -> None:
    tuner = AggregatorTuner()

    upper_metric_data = {
        "fidelity": _iterations(1.0),
        "utility": _iterations(1.0),
        "privacy": _iterations(0.0),
    }
    lower_metric_data = {
        "fidelity": _iterations(0.0),
        "utility": _iterations(0.0),
        "privacy": _iterations(1.0),
    }

    results = tuner.tune_from_metrics(upper_metric_data, lower_metric_data)
    composite_name = results["optimal"]["composite"]
    composite_func = tuner.aggregation_functions[composite_name]

    assert composite_name != "median"
    assert 0.0 < composite_func([1.0, 1.0, 0.0]) < 1.0
    assert 0.0 < composite_func([0.0, 0.0, 1.0]) < 1.0


# ── Calibration uses the evaluation's hierarchy (review, coordinator item 1) ──

# 3 marginal metrics (2 subcategories), 1 global: flat and hierarchical differ
_UPPER_ITER = {
    "fidelity.ks": 0.9,
    "fidelity.wasserstein": 0.9,
    "fidelity.tvd": 0.9,
    "fidelity.mmd": 0.3,
}


def _evaluation_family_score(values, agg_name, weights=None, risk_aversion=DEFAULT_RISK_AVERSION):
    import json
    import tempfile
    from pathlib import Path

    from metis.application.aggregator import Aggregator
    from metis.domain.entities import EvalPlan, MetricResult

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.json"
        path.write_text(json.dumps({"optimal": {"fidelity": agg_name, "composite": "mean"}}))
        agg = Aggregator(risk_aversion=risk_aversion, optimal_aggregators_path=str(path))
    results = [
        MetricResult(id=k, value=v, details={}, family="fidelity") for k, v in values.items()
    ]
    return agg.aggregate(results, EvalPlan(metric_ids=list(values)), weights)["fidelity_score"]


def test_reaggregate_matches_the_hierarchical_evaluation() -> None:
    import pytest

    for name in ("mean", "median", "ssd"):
        (score,) = AggregatorTuner.reaggregate({"fidelity": [_UPPER_ITER]}, {"fidelity": name})[
            "fidelity"
        ]
        assert score == pytest.approx(_evaluation_family_score(_UPPER_ITER, name))
    # the flat mean (0.75) is not what the evaluation computes (mean of 0.9, 0.3)
    (flat_was,) = AggregatorTuner.reaggregate({"fidelity": [_UPPER_ITER]}, {"fidelity": "mean"})[
        "fidelity"
    ]
    assert flat_was == pytest.approx(0.6)


def test_reaggregate_uses_the_configured_weights() -> None:
    import pytest

    from metis.calibrate.optimization.aggregator_tuner import HierarchicalScorer

    cfg = {"aggregation": {"weights": {"fidelity.global": 0, "utility": 0}, "risk_aversion": 3}}
    scorer = HierarchicalScorer.from_config(cfg)
    (score,) = AggregatorTuner.reaggregate(
        {"fidelity": [_UPPER_ITER]}, {"fidelity": "ssd"}, scorer
    )["fidelity"]
    expected = _evaluation_family_score(
        _UPPER_ITER, "ssd", weights={"fidelity.global": 0}, risk_aversion=3
    )
    assert score == pytest.approx(expected)
    assert score == pytest.approx(0.9)  # only the marginal category is left


def test_fidelity_aggregator_cannot_win_by_reversing_the_references() -> None:
    tuner = AggregatorTuner()
    upper = _iterations(0.1)  # real-vs-real scored *below* noise
    lower = _iterations(0.9)
    _, fid_error = tuner._pick_best_family_aggregator(upper, lower, "fidelity")
    _, priv_error = tuner._pick_best_family_aggregator(upper, lower, "privacy")
    assert fid_error > 1.0  # before: min(normal, inverted) gave ~0.02
    assert priv_error < 0.05  # privacy keeps the orientation-agnostic error


def test_candidate_aggregators_are_distinct_functions() -> None:
    import itertools

    import numpy as np

    tuner = AggregatorTuner()
    assert "fsd" not in tuner.available_aggregators  # fsd is the mean: it could never win
    rng = np.random.default_rng(0)
    vectors = [list(rng.uniform(size=n)) for n in (3, 7, 12, 25)]
    for a, b in itertools.combinations(tuner.available_aggregators, 2):
        fa, fb = tuner.aggregation_functions[a], tuner.aggregation_functions[b]
        assert any(abs(fa(v) - fb(v)) > 1e-9 for v in vectors), (a, b)
