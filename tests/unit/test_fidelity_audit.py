"""Fidelity fixes from the 2026-10-08 audit (F1, F4-F9, F12, ordinal shadows)."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from metis.infrastructure.metrics.fidelity.conditional.cat_cat.chi2_stat import (
    Chi2StatMetric,
    bias_corrected_v2,
)
from metis.infrastructure.metrics.fidelity.conditional.cat_cat.cramers_v import CramersVMetric
from metis.infrastructure.metrics.fidelity.conditional.cat_cat.theils_u import TheilsUMetric
from metis.infrastructure.metrics.fidelity.conditional.num_cat.eta_squared import EtaSquaredMetric
from metis.infrastructure.metrics.fidelity.conditional.num_num.dcor import DistanceCorrelationMetric
from metis.infrastructure.metrics.fidelity.conditional.num_num.mi import MutualInformationMetric
from metis.infrastructure.metrics.fidelity.conditional.num_num.pearson import (
    PearsonCorrelationMetric,
)
from metis.infrastructure.metrics.fidelity.conditional.num_num.spearman import (
    SpearmanCorrelationMetric,
)
from metis.infrastructure.metrics.fidelity.conditional.num_num.correlation_matrix import (
    CorrelationMatrixMetric,
)
from metis.infrastructure.metrics.fidelity.global_metrics.distances.energy_distance import (
    EnergyDistanceMetric,
)
from metis.infrastructure.metrics.fidelity.global_metrics.distances.mmd import MMDMetric
from metis.infrastructure.metrics.fidelity.marginal.tails.outliers_coverage import (
    OutliersCoverageMetric,
)
from metis.infrastructure.metrics.fidelity.marginal.coverage.kl import KLDivergenceMetric
from metis.infrastructure.metrics.fidelity.marginal.coverage.psi import PSIMetric
from metis.infrastructure.metrics.fidelity.marginal.coverage.tvd import TVDMetric
from metis.infrastructure.metrics.fidelity.marginal.tails.anderson_darling import (
    AndersonDarlingMetric,
    ad_discrepancy,
)
from metis.infrastructure.metrics.fidelity.marginal.tails.kde_ise import KDEISEMetric
from metis.infrastructure.metrics.fidelity.marginal.tails.wasserstein import WassersteinMetric
from metis.infrastructure.runtime.cache import StatsStore
from metis.shared.normalization import clamp, normalize_metric_value

UNBOUNDED_COLUMN_METRICS = [
    WassersteinMetric,
    KDEISEMetric,
    AndersonDarlingMetric,
    KLDivergenceMetric,
    PSIMetric,
]


def _numeric(n: int, k: int, seed: int, shift: float = 0.0, scale: float = 3.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k))
    z[:, 1:] = 0.6 * z[:, :1] + 0.8 * z[:, 1:]
    return pd.DataFrame((z + shift) * scale + 10.0, columns=[f"x{i}" for i in range(k)])


def _score(metric_cls, real, synth, context=None, **kwargs) -> float:
    return metric_cls(**kwargs).fit(real, synth, context or {}).compute().value


# --------------------------------------------------------------------------- F1


@pytest.mark.parametrize("metric_cls", UNBOUNDED_COLUMN_METRICS)
def test_f1_exact_copy_scores_one(metric_cls) -> None:
    real = _numeric(400, 8, 0)
    assert _score(metric_cls, real, real.copy()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("metric_cls", UNBOUNDED_COLUMN_METRICS)
def test_f1_all_columns_shifted_5_sigma_scores_near_zero(metric_cls) -> None:
    # Old min-max-between-columns normalisation gave 0.5 here (all equally bad).
    real = _numeric(400, 8, 0)
    synth = _numeric(400, 8, 0, shift=5.0)
    assert _score(metric_cls, real, synth) < 0.05


@pytest.mark.parametrize("metric_cls", UNBOUNDED_COLUMN_METRICS)
def test_f1_same_process_is_high_and_stable_in_number_of_columns(metric_cls) -> None:
    v8 = _score(metric_cls, _numeric(400, 8, 0), _numeric(400, 8, 1))
    v16 = _score(metric_cls, _numeric(400, 16, 0), _numeric(400, 16, 1))
    assert v8 > 0.75
    assert v16 > 0.75
    assert abs(v8 - v16) < 0.1


def test_f1_column_score_does_not_depend_on_other_columns() -> None:
    real = _numeric(400, 4, 0)
    synth = _numeric(400, 4, 1)
    alone = WassersteinMetric().fit(real[["x0"]], synth[["x0"]], {}).compute()
    synth_bad = synth.copy()
    synth_bad["x3"] += 100.0
    together = WassersteinMetric().fit(real, synth_bad, {}).compute()
    assert together.details["x0"]["normalized_value"] == pytest.approx(
        alone.details["x0"]["normalized_value"]
    )


def test_f1_wasserstein_is_scale_free() -> None:
    real = _numeric(400, 3, 0)
    synth = _numeric(400, 3, 1)
    assert _score(WassersteinMetric, real, synth) == pytest.approx(
        _score(WassersteinMetric, real * 1000, synth * 1000)
    )


def test_anderson_darling_detects_a_shift() -> None:
    # It used an invalid scipy variant, raised on every call and scored 1.0.
    real = _numeric(400, 2, 0)
    assert _score(AndersonDarlingMetric, real, real + 3.0) < 0.8  # 1 sigma shift
    assert _score(AndersonDarlingMetric, real, real + 30.0) < 0.01  # no overlap


def test_anderson_darling_does_not_depend_on_n() -> None:
    # Before (statistic of the test, exp(-A2/5)): 0.82 / 0.39 / 0.003 for the
    # same shift with n = 100 / 400 / 2000.
    scores = {}
    for n in (100, 400, 2000):
        reps = [
            _score(AndersonDarlingMetric, _numeric(n, 2, s), _numeric(n, 2, 100 + s, shift=0.5))
            for s in range(5)
        ]
        scores[n] = float(np.mean(reps))
    assert max(scores.values()) - min(scores.values()) < 0.04
    assert 0.85 < scores[2000] < 0.97


def test_ad_discrepancy_is_symmetric_and_bounded() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=300), rng.normal(0.4, size=120)
    assert ad_discrepancy(x, y) == pytest.approx(ad_discrepancy(y, x))
    assert ad_discrepancy(x, x) == 0.0
    assert ad_discrepancy(x, x + 100) == pytest.approx(1.0, abs=0.01)
    assert ad_discrepancy(np.zeros(10), np.ones(10)) == 1.0


# --------------------------------------------------------------------------- F4


def test_f4_context_reaches_every_fidelity_base_class() -> None:
    real = _numeric(100, 3, 0)
    ctx = {"seed": 7, "stats_store": StatsStore()}
    for metric in (WassersteinMetric(), TVDMetric(), PearsonCorrelationMetric(), MMDMetric()):
        metric.fit(real, real, ctx)
        assert metric._context is ctx
        assert metric.seed == 7


def test_f4_seed_changes_subsampled_mmd() -> None:
    real = _numeric(600, 3, 0)
    synth = _numeric(600, 3, 1, shift=0.5)
    a = MMDMetric(max_samples=100).fit(real, synth, {"seed": 1}).compute()
    b = MMDMetric(max_samples=100).fit(real, synth, {"seed": 2}).compute()
    c = MMDMetric(max_samples=100).fit(real, synth, {"seed": 1}).compute()
    assert a.details["raw_mmd"] != b.details["raw_mmd"]
    assert a.details["raw_mmd"] == c.details["raw_mmd"]


def test_f4_stats_store_shares_contingency_tables() -> None:
    rng = np.random.default_rng(0)
    real = pd.DataFrame({c: rng.choice(list("abc"), 200) for c in "pqr"})
    synth = pd.DataFrame({c: rng.choice(list("abc"), 200) for c in "pqr"})
    store = StatsStore()
    ctx = {"stats_store": store, "seed": 0}
    for cls in (CramersVMetric, TheilsUMetric, Chi2StatMetric):
        cls().fit(real, synth, ctx).compute()
    # 3 pairs x 2 tables built once, then reused by the two other metrics;
    # Theil's U reads each one twice (both directions, the second transposed).
    assert store._miss_count == 6
    assert store._hit_count == 18


# --------------------------------------------------------------------------- F5


def _rv_details(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    return CorrelationMatrixMetric().fit(real, synth, {}).compute().details


def _blocks(n: int, seed: int, scale: float = 1.0) -> pd.DataFrame:
    k = 8
    corr = np.full((k, k), 0.15)
    corr[:4, :4] = 0.6
    corr[4:, 4:] = 0.6
    corr = scale * corr + (1 - scale) * np.eye(k)
    np.fill_diagonal(corr, 1.0)
    z = np.random.default_rng(seed).standard_normal((n, k)) @ np.linalg.cholesky(corr).T
    return pd.DataFrame(z, columns=[f"x{i}" for i in range(k)])


def test_f5_rv_is_undefined_without_structure_in_the_real_table() -> None:
    # Before: a noise ridge made it ~0.87 +/- 0.16. Without structure the
    # pattern of correlations is 0/0: NaN, and the combined score falls back to
    # Frobenius + spectral with renormalized weights.
    real = pd.DataFrame(np.random.default_rng(0).standard_normal((400, 8)))
    synth = pd.DataFrame(np.random.default_rng(1).standard_normal((400, 8)))
    result = CorrelationMatrixMetric().fit(real, synth, {}).compute()
    assert math.isnan(result.details["rv_coefficient"])
    assert result.details["rv_defined"] is False
    expected = (
        0.4 * result.details["frobenius_score"] + 0.3 * result.details["spectral_score"]
    ) / 0.7
    assert result.value == pytest.approx(expected)


@pytest.mark.parametrize("n", [100, 400, 4000])
def test_f5_rv_of_destroyed_structure_is_zero_for_any_n(n: int) -> None:
    real = _blocks(n, 0)
    rng = np.random.default_rng(1)
    synth = _blocks(n, 2).apply(lambda c: rng.permutation(c.to_numpy()))
    # 0 unless the sphericity test gives a false positive on the synthetic (5 %).
    assert _rv_details(real, synth)["rv_coefficient"] < 0.25


@pytest.mark.parametrize("n", [100, 400, 4000])
def test_f5_rv_of_same_process_is_near_one(n: int) -> None:
    assert _rv_details(_blocks(n, 0), _blocks(n, 1))["rv_coefficient"] > 0.9


def test_f5_rv_ignores_uniform_weakening_but_frobenius_does_not() -> None:
    real, weak = _blocks(4000, 0), _blocks(4000, 1, scale=0.5)
    details = _rv_details(real, weak)
    assert details["rv_coefficient"] > 0.95
    assert details["frobenius_score"] < _rv_details(real, _blocks(4000, 1))["frobenius_score"]


def test_f5_rv_detects_a_different_pattern() -> None:
    real = _blocks(4000, 0)
    other = _blocks(4000, 1)[[f"x{i}" for i in (0, 4, 1, 5, 2, 6, 3, 7)]]
    other.columns = real.columns
    assert _rv_details(real, other)["rv_coefficient"] < 0.75


# --------------------------------------------------------------------------- F6 / F7


def test_f6_pair_delta_uses_full_metric_id() -> None:
    metric = Chi2StatMetric()
    assert metric._normalize_pair_delta(0.3) == pytest.approx(
        normalize_metric_value("fidelity.chi2_stat", 0.3)
    )
    # chi2_stat compares bias-corrected V^2 (bounded): 1 - delta, no scale.
    assert metric._normalize_pair_delta(0.3) == pytest.approx(0.7)


def _cat_pair(n: int, r: int, c: int, strength: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, r, n)
    y = np.where(rng.random(n) < strength, x % c, rng.integers(0, c, n))
    return pd.DataFrame({"a": [f"a{v}" for v in x], "b": [f"b{v}" for v in y]})


def test_bias_corrected_v2_matches_cramers_v_squared_without_bias() -> None:
    table = np.array([[30, 10], [10, 30]])
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    expected = max(0.0, chi2 / n - 1 / (n - 1)) / (2 - 1 / (n - 1) - 1)
    assert bias_corrected_v2(table) == pytest.approx(expected)
    assert bias_corrected_v2(np.array([[5, 0], [0, 5]])) <= 1.0
    assert bias_corrected_v2(np.array([[3, 4, 5]])) == 0.0


@pytest.mark.parametrize(("r", "c"), [(2, 2), (3, 4), (6, 8)])
def test_chi2_stat_loss_of_association_does_not_depend_on_cardinality(r: int, c: int) -> None:
    # Before: exp(-|d phi^2| / 0.3) gave 0.30 in 2x2 and 0.001 in 6x8 for the
    # same destroyed association (phi^2 grows with min(r, c) - 1).
    real = _cat_pair(2000, r, c, 0.6, 0)
    synth = _cat_pair(2000, r, c, 0.6, 1)
    synth["b"] = np.random.default_rng(2).permutation(synth["b"].to_numpy())
    score = _score(Chi2StatMetric, real, synth)
    assert 0.55 < score < 0.70  # 1 - V^2 with V ~ 0.6


@pytest.mark.parametrize(("nr", "ns"), [(2000, 200), (200, 2000)])
def test_chi2_stat_of_independent_columns_does_not_depend_on_n(nr: int, ns: int) -> None:
    scores = [
        _score(Chi2StatMetric, _cat_pair(nr, 6, 8, 0.0, s), _cat_pair(ns, 6, 8, 0.0, 100 + s))
        for s in range(10)
    ]
    assert np.mean(scores) > 0.98


@pytest.mark.parametrize(
    ("metric_cls", "raw_key"),
    [(MMDMetric, "raw_mmd"), (EnergyDistanceMetric, "energy_coefficient")],
)
def test_f7_global_distances_use_the_central_map(metric_cls, raw_key) -> None:
    real = _numeric(300, 4, 0)
    synth = _numeric(300, 4, 1, shift=0.5)
    result = metric_cls().fit(real, synth, {}).compute()
    raw = result.details[raw_key]
    assert result.value == pytest.approx(normalize_metric_value(f"fidelity.{metric_cls.name}", raw))


# --------------------------------------------------------------------------- F8


def test_f8_mi_is_bias_corrected_for_independent_columns() -> None:
    metric = MutualInformationMetric()
    rng = np.random.default_rng(0)
    values = [
        metric._compute_mi(rng.standard_normal(400), rng.standard_normal(400)) for _ in range(30)
    ]
    # Plug-in MI on 10x10 bins is ~0.1 nats here.
    assert np.mean(values) < 0.03


def test_f8_mi_uses_bin_edges_fitted_on_the_real_table() -> None:
    real = _numeric(400, 2, 0)
    synth = real + 1000.0  # same dependence, but outside the real grid
    result = MutualInformationMetric().fit(real, synth, {}).compute()
    pair = result.details["x0|x1"]
    assert pair["real_value"] > 0.2
    assert pair["synth_value"] == pytest.approx(0.0)


def test_f8_outliers_coverage_same_process_is_high() -> None:
    values = [
        _score(
            OutliersCoverageMetric,
            pd.DataFrame(np.random.default_rng(s).standard_normal((400, 8))),
            pd.DataFrame(np.random.default_rng(100 + s).standard_normal((400, 8))),
        )
        for s in range(10)
    ]
    assert np.mean(values) > 0.8


def test_f8_outliers_coverage_uses_real_fences() -> None:
    real = pd.DataFrame(np.random.default_rng(0).standard_normal((400, 4)))
    result = OutliersCoverageMetric().fit(real, real + 5.0, {}).compute()
    assert result.details["outlier_overlap"] < 0.1
    assert result.value < 0.1


# --------------------------------------------------------------------------- F9


def test_f9_clamp_propagates_nan() -> None:
    assert math.isnan(clamp(float("nan")))
    assert clamp(float("inf")) == 1.0
    assert clamp(float("-inf")) == 0.0


def test_f9_column_metric_without_valid_columns_is_nan_and_not_applicable() -> None:
    real = pd.DataFrame({"c": list("ab" * 20)})
    result = WassersteinMetric().fit(real, real, {}).compute()
    assert math.isnan(result.value)
    assert result.status == "not_applicable"
    assert "reason" in result.details


def test_f9_pair_metric_without_valid_pairs_is_not_applicable() -> None:
    real = pd.DataFrame({"x": np.arange(40.0)})
    result = PearsonCorrelationMetric().fit(real, real, {}).compute()
    assert math.isnan(result.value)
    assert result.status == "not_applicable"
    assert "reason" in result.details


# --------------------------------------------------------------------------- ordinals


def test_ordinal_shadow_is_not_counted_twice_in_full() -> None:
    rng = np.random.default_rng(0)
    q = rng.choice(["low", "mid", "high"], 200)
    full = pd.DataFrame(
        {
            "q": q,
            "q__ord": pd.Series(q).map({"low": 0.0, "mid": 0.5, "high": 1.0}).to_numpy(),
            "x": rng.standard_normal(200),
        }
    )
    tvd = TVDMetric().fit(full, full, {})
    assert sorted(tvd._columns) == ["q", "x"]
    eta = EtaSquaredMetric().fit(full, full, {})
    assert eta._pairs == [("x", "q")]
    # In the numeric slice (no categorical twin) the shadow is kept.
    num_only = WassersteinMetric().fit(full[["q__ord", "x"]], full[["q__ord", "x"]], {})
    assert sorted(num_only._columns) == ["q__ord", "x"]


# --------------------------------------------------------------------------- F12


def _with_nans() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.standard_normal((300, 4)), columns=list("abcd"))
    a["b"] += a["a"] ** 2
    b = pd.DataFrame(rng.standard_normal((300, 4)), columns=list("abcd"))
    a.iloc[::7, 1] = np.nan
    b.iloc[::5, 2] = np.nan
    return a, b


@pytest.mark.parametrize(
    ("metric_cls", "fn"),
    [
        (PearsonCorrelationMetric, lambda x, y: stats.pearsonr(x, y)[0]),
        (SpearmanCorrelationMetric, lambda x, y: stats.spearmanr(x, y)[0]),
        (
            DistanceCorrelationMetric,
            lambda x, y: DistanceCorrelationMetric()._compute_dcor(x, y),
        ),
    ],
)
def test_f12_fast_paths_equal_the_per_pair_computation(metric_cls, fn) -> None:
    real, synth = _with_nans()
    result = metric_cls().fit(real, synth, {}).compute()
    for key in ("a|b", "b|c", "c|d"):
        c1, c2 = key.split("|")
        r = real[[c1, c2]].dropna()
        s = synth[[c1, c2]].dropna()
        assert result.details[key]["real_value"] == pytest.approx(fn(r[c1].values, r[c2].values))
        assert result.details[key]["synth_value"] == pytest.approx(fn(s[c1].values, s[c2].values))


def test_f12_contingency_equals_crosstab() -> None:
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"p": rng.choice(list("xyz"), 150), "q": rng.choice(list("ab"), 150)})
    metric = CramersVMetric().fit(real, real, {})
    table = metric._contingency("real", real["p"], real["q"])
    np.testing.assert_array_equal(table, pd.crosstab(real["p"], real["q"]).values)
