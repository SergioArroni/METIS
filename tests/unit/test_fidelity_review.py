"""Second review of the fidelity metrics (R-F findings, 2026-10-09)."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency, kruskal

from metis.infrastructure.metrics.fidelity.conditional.cat_cat.chi2_stat import bias_corrected_v2
from metis.infrastructure.metrics.fidelity.conditional.cat_cat.cramers_v import CramersVMetric
from metis.infrastructure.metrics.fidelity.conditional.num_cat.kruskal_epsilon import (
    KruskalEpsilonMetric,
)
from metis.infrastructure.metrics.fidelity.global_metrics.distances.energy_distance import (
    EnergyDistanceMetric,
)
from metis.infrastructure.metrics.fidelity.global_metrics.distances.mmd import MMDMetric
from metis.infrastructure.metrics.fidelity.marginal.tails.outliers_coverage import (
    OutliersCoverageMetric,
)
from metis.infrastructure.metrics.fidelity.marginal.coverage.entropy_delta import (
    ShannonEntropyDeltaMetric,
)
from metis.infrastructure.metrics.fidelity.marginal.coverage.js import JSDivergenceMetric
from metis.infrastructure.metrics.fidelity.marginal.coverage.tvd import TVDMetric
from metis.infrastructure.metrics.fidelity.marginal.scales.cohens_d import CohensD
from metis.infrastructure.metrics.fidelity.marginal.scales.delta_iqr import DeltaIQRMetric
from metis.infrastructure.metrics.fidelity.marginal.scales.delta_mad import DeltaMADMetric
from metis.infrastructure.metrics.fidelity.marginal.scales.delta_mean import DeltaMeanMetric
from metis.infrastructure.metrics.fidelity.marginal.scales.delta_median import DeltaMedianMetric
from metis.infrastructure.metrics.fidelity.marginal.tails.hellinger import HellingerMetric
from metis.infrastructure.metrics.fidelity.marginal.tails.kde_ise import KDEISEMetric
from metis.shared.multi_run_stats import calculate_multi_run_statistics

BINNED = [TVDMetric, JSDivergenceMetric, HellingerMetric]


def _score(metric_cls, real, synth, **kwargs) -> float:
    return metric_cls(**kwargs).fit(real, synth, {}).compute().value


def _normal(n: int, seed: int, loc: float = 0.0, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(loc, scale, n)})


# ------------------------------------------------- binned divergences (TVD, JS, Hellinger)


@pytest.mark.parametrize("metric_cls", BINNED)
def test_one_synthetic_outlier_does_not_improve_the_score(metric_cls) -> None:
    # Pooled 50 equal-width bins: a single synthetic value at 1e4 stretched the
    # grid, both tables fell in the same bin and the score jumped to ~1.
    real, synth = _normal(400, 0), _normal(400, 1)
    with_outlier = synth.copy()
    with_outlier.loc[0, "x"] = 1e4
    clean = _score(metric_cls, real, synth)
    dirty = _score(metric_cls, real, with_outlier)
    assert dirty <= clean + 0.01


@pytest.mark.parametrize("metric_cls", BINNED)
def test_bin_edges_do_not_depend_on_the_synthetic_table(metric_cls) -> None:
    # Moving synthetic mass far outside the real range only touches the outer bin.
    real, synth = _normal(400, 0), _normal(400, 1)
    far = synth.copy()
    far.loc[:9, "x"] = 50.0
    farther = synth.copy()
    farther.loc[:9, "x"] = 5000.0
    assert _score(metric_cls, real, far) == pytest.approx(_score(metric_cls, real, farther))


@pytest.mark.parametrize("metric_cls", BINNED)
def test_same_process_score_is_high_and_stable_in_n(metric_cls) -> None:
    # Pooled 50 bins: ~0.75 at n=200 and ~0.95 at n=5000 (plug-in bias of 50
    # bins). Real deciles: ~0.88 and ~0.98. Some n-dependence remains (any
    # plug-in estimator has it), but it is halved.
    def mean_score(n: int) -> float:
        return float(
            np.mean(
                [_score(metric_cls, _normal(n, 2 * k), _normal(n, 2 * k + 1)) for k in range(6)]
            )
        )

    small, large = mean_score(200), mean_score(5000)
    assert small > 0.85
    assert large - small < 0.12


@pytest.mark.parametrize("metric_cls", BINNED)
def test_a_one_sigma_shift_is_still_detected(metric_cls) -> None:
    same = _score(metric_cls, _normal(400, 0), _normal(400, 1))
    shifted = _score(metric_cls, _normal(400, 0), _normal(400, 1, loc=1.0))
    assert shifted < same - 0.2


# ---------------------------------------------------------------- entropy delta


def test_entropy_delta_survives_one_synthetic_outlier() -> None:
    real, synth = _normal(400, 0), _normal(400, 1)
    with_outlier = synth.copy()
    with_outlier.loc[0, "x"] = 1e4
    assert _score(ShannonEntropyDeltaMetric, real, with_outlier) > 0.9


def test_entropy_delta_detects_a_wider_synthetic_column() -> None:
    real = _normal(400, 0)
    assert _score(ShannonEntropyDeltaMetric, real, _normal(400, 1)) > 0.95
    assert _score(ShannonEntropyDeltaMetric, real, _normal(400, 1, scale=3.0)) < 0.85


# -------------------------------------------------------- scales: unit-free deltas


@pytest.mark.parametrize("metric_cls", [DeltaMeanMetric, DeltaMedianMetric])
def test_location_delta_does_not_depend_on_the_unit_when_real_scale_is_zero(metric_cls) -> None:
    rng = np.random.default_rng(0)
    zero_inflated = np.where(rng.random(400) < 0.8, 0.0, rng.exponential(1.0, 400))
    other = np.where(rng.random(400) < 0.3, 0.0, rng.exponential(1.0, 400)) + 0.5
    scores = []
    for unit in (1e-3, 1.0, 1e3):
        real = pd.DataFrame(
            {"x": np.zeros(400) if metric_cls is DeltaMeanMetric else zero_inflated}
        )
        scores.append(_score(metric_cls, real * unit, pd.DataFrame({"x": other}) * unit))
    assert scores == pytest.approx([scores[0]] * 3)


@pytest.mark.parametrize("metric_cls", [DeltaIQRMetric, DeltaMADMetric])
def test_dispersion_delta_does_not_depend_on_the_unit_when_real_scale_is_zero(metric_cls) -> None:
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": np.where(rng.random(400) < 0.8, 0.0, rng.exponential(1.0, 400))})
    synth = pd.DataFrame({"x": rng.exponential(1.0, 400)})
    scores = [_score(metric_cls, real * u, synth * u) for u in (1e-3, 1.0, 1e3)]
    assert scores == pytest.approx([scores[0]] * 3)
    assert scores[0] < 0.9


def test_cohens_d_of_two_different_constants_is_the_worst_score() -> None:
    real = pd.DataFrame({"x": np.full(50, 1.0)})
    assert _score(CohensD, real, pd.DataFrame({"x": np.full(50, 5.0)})) == 0.0
    assert _score(CohensD, real, real.copy()) == 1.0


def test_kde_ise_penalises_a_synthetic_column_collapsed_to_the_mean() -> None:
    real = _normal(400, 0)
    collapsed = pd.DataFrame({"x": np.full(400, float(real["x"].mean()))})
    assert _score(KDEISEMetric, real, collapsed) == 0.0


def test_cramers_v_uses_the_uncorrected_chi2() -> None:
    rng = np.random.default_rng(0)
    p = rng.choice(["a", "b"], 60)
    q = np.where(rng.random(60) < 0.7, p, rng.choice(["a", "b"], 60))
    table = pd.crosstab(p, q).values
    # Uncorrected chi^2 (no Yates), then Bergsma's bias correction (2026-10-19).
    plug_in = math.sqrt(chi2_contingency(table, correction=False)[0] / 60)
    assert CramersVMetric()._cramers_v(table) == pytest.approx(math.sqrt(bias_corrected_v2(table)))
    assert CramersVMetric()._cramers_v(table) < plug_in


def test_kruskal_epsilon_counts_only_the_rows_in_the_groups() -> None:
    rng = np.random.default_rng(0)
    g = np.array(["a"] * 30 + ["b"] * 30 + ["c"])  # "c" is a singleton, dropped
    v = rng.standard_normal(61) + (g == "a")
    real = pd.DataFrame({"v": v, "g": g})
    metric = KruskalEpsilonMetric().fit(real, real, {})
    r_eps, _ = metric._compute_pair(real["v"], real["g"], real["v"], real["g"])
    h = kruskal(v[g == "a"], v[g == "b"])[0]
    assert r_eps == pytest.approx(h / (60 - 1))


# --------------------------------------------------------------- global metrics


@pytest.mark.parametrize("metric_cls", [MMDMetric, EnergyDistanceMetric, OutliersCoverageMetric])
def test_global_metrics_see_the_ordinal_once_through_its_rank(metric_cls) -> None:
    """Numeric-only global metrics measure the ordinal by its rank ``__ord``.

    They used to drop it (its categorical twin is not numeric, so the
    ordinal was ignored altogether): a synthetic table that inverted the
    ordinal scored exactly as the real one.
    """
    rng = np.random.default_rng(0)
    real = pd.DataFrame(
        {
            "x": rng.standard_normal(200),
            "q": pd.Categorical(rng.choice(["lo", "hi"], 200)),
        }
    )
    real["q__ord"] = (real["q"] == "hi").astype(float)
    synth = real.copy()
    synth["q__ord"] = 0.5 * synth["q__ord"]  # only the rank differs (level squeezed)
    result = metric_cls().fit(real, synth, {}).compute()
    same = metric_cls().fit(real, real, {}).compute()
    assert result.details["n_columns" if "n_columns" in result.details else "n_features"] == 2
    if metric_cls is not OutliersCoverageMetric:  # a binary rank has no outliers
        assert result.value < same.value


# ------------------------------------------------------------------- multi-run


def test_multi_run_statistics_skip_nan_scores_and_use_every_run() -> None:
    runs = [
        {"scores": {"fidelity_score": 0.8, "utility_score": float("nan")}},
        {"scores": {"fidelity_score": 0.6, "utility_score": 0.4, "privacy_score": 0.9}},
        {"scores": {"fidelity_score": 0.7, "utility_score": 0.5, "privacy_score": 0.7}},
    ]
    stats = calculate_multi_run_statistics(runs)
    assert stats["fidelity_score"]["mean"] == pytest.approx(0.7)
    assert stats["utility_score"]["mean"] == pytest.approx(0.45)
    assert stats["utility_score"]["n_runs"] == 2
    assert stats["privacy_score"]["n_runs"] == 2


def test_column_metrics_accept_non_string_column_names() -> None:
    # ``_ordinal_shadows`` called ``str.endswith`` on every name: integer
    # column names raised and every global metric returned NaN.
    rng = np.random.default_rng(0)
    real = pd.DataFrame(rng.standard_normal((200, 3)))
    assert _score(OutliersCoverageMetric, real, real.copy()) == pytest.approx(1.0)
    assert _score(TVDMetric, real, real.copy()) == pytest.approx(1.0)
