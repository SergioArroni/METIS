"""Method fixes of 2026-10-19: utility needs a holdout; δ ≥ 1/n voids a DP
guarantee (Dwork & Roth 2014); association measures bias-corrected; energy
score free of the number of columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from metis.infrastructure.metrics.fidelity.conditional.cat_cat.cramers_v import CramersVMetric
from metis.infrastructure.metrics.fidelity.conditional.num_cat.eta_squared import EtaSquaredMetric
from metis.infrastructure.metrics.fidelity.conditional.num_num.dcor import DistanceCorrelationMetric
from metis.infrastructure.metrics.fidelity.global_metrics.distances.energy_distance import (
    EnergyDistanceMetric,
)
from metis.infrastructure.metrics.privacy import DifferentialPrivacyMetric
from metis.infrastructure.metrics.utility import ml_efficiency_metric as mem
from metis.infrastructure.metrics.utility.ml_efficiency.base import StrategyResult, TrainingStrategy
from metis.interface.sdk import Evaluator

# ── utility without holdout ──────────────────────────────────────────────


def test_utility_without_holdout_does_not_apply():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    real = pd.DataFrame({"x": x, "y": np.where(x > 0, "a", "b")})
    copy = real.sample(frac=1, random_state=1).reset_index(drop=True)
    cfg = {
        "data": {"schema": {"x": "continuous", "y": "nominal"}, "target": "y"},
        "metrics": ["utility.ml_efficiency"],
        "evaluation": {"n_runs": 1},
    }
    (result,) = Evaluator().evaluate(real, copy, cfg, seed=0).results
    assert result.status == "not_applicable"
    assert "holdout" in result.details["reason"]


# ── δ as a validity condition ────────────────────────────────────────────


def _dp(delta: float, n: int = 1000):
    data = pd.DataFrame({"x": np.arange(float(n))})
    return (
        DifferentialPrivacyMetric(epsilon=1.0, delta=delta).fit(data, data, {"seed": 0}).compute()
    )


def test_a_small_delta_leaves_the_epsilon_score():
    result = _dp(1e-5)
    assert result.value == pytest.approx(np.exp(-0.5))
    assert result.details["delta_valid"] is True


def test_delta_at_least_one_over_n_voids_the_guarantee():
    result = _dp(1e-3, n=1000)
    assert result.value == 0.0
    assert result.details["delta_valid"] is False
    assert result.details["delta_limit_1_over_n"] == pytest.approx(1e-3)


# ── bias-corrected association ───────────────────────────────────────────


@pytest.mark.parametrize("n", [60, 400])
def test_independent_columns_score_no_association_on_average(n):
    """Corrected estimates are ≈ 0 on average for unrelated columns at any n
    (one table still scatters); the plug-in ones are clearly positive with few rows."""
    rng = np.random.default_rng(n)
    v, v_plug, omega, eta, dcor = [], [], [], [], []
    for _ in range(100):
        a, b = rng.choice(list("abcd"), n), rng.choice(list("abc"), n)
        table = pd.crosstab(a, b).values
        v.append(CramersVMetric()._cramers_v(table))
        v_plug.append(np.sqrt(chi2_contingency(table, correction=False)[0] / (n * 2)))
        groups = [rng.normal(size=n // 4) for _ in range(4)]
        omega.append(EtaSquaredMetric()._compute_eta_squared(groups))
        allv = np.concatenate(groups)
        eta.append(
            sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
            / ((allv - allv.mean()) ** 2).sum()
        )
        if len(dcor) < 20:
            dcor.append(
                DistanceCorrelationMetric()._compute_dcor(rng.normal(size=n), rng.normal(size=n))
            )
    assert np.mean(v) < 0.5 * np.mean(v_plug)
    assert np.mean(omega) < 0.5 * np.mean(eta)  # clipping at 0 leaves a small positive rest
    assert np.mean(dcor) < 0.06


# ── energy free of the number of columns ─────────────────────────────────


def test_energy_score_does_not_fall_with_more_columns():
    rng = np.random.default_rng(0)
    scores = []
    for d in (5, 40):
        cols = [f"c{i}" for i in range(d)]
        real = pd.DataFrame(rng.normal(size=(800, d)), columns=cols)
        shifted = pd.DataFrame(rng.normal(0.3, 1, size=(800, d)), columns=cols)
        scores.append(EnergyDistanceMetric().fit(real, shifted, {}).compute().value)
    assert abs(scores[0] - scores[1]) < 0.02


# ── symmetric utility ────────────────────────────────────────────────────


def test_utility_above_ttr_costs_as_much_as_below():
    scale = {"lower": 0.05, "upper": 0.25, "ttr_gain": 0.30}  # tolerance 0.05 each side

    def utility(score: float) -> float:
        result = StrategyResult(
            strategy=TrainingStrategy.TSTR,
            score=score,
            raw_metric=score,
            scores_per_run=[score],
            std=0.0,
            best_params={},
            is_valid=True,
        )
        return mem._absolute_utility(result, [0.5], scale, "classification")[0]

    # AUC 0.80 is the TTR gain (0.30 over 0.5): within the noise, 1.
    assert utility(0.80) == pytest.approx(1.0)
    assert utility(0.70) == pytest.approx(utility(0.90))  # 0.10 below == 0.10 above
    assert utility(0.70) == pytest.approx((0.20 - 0.05) / 0.20)  # the one-sided value below


# ── real and synthetic at the same size ──────────────────────────────────


def test_real_and_synthetic_are_compared_at_the_size_of_the_smaller():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=3000)})
    synth = pd.DataFrame({"x": rng.normal(size=2500)})
    cfg = {"data": {"schema": {"x": "continuous"}}, "metrics": ["fidelity.ks"]}
    summary = Evaluator().evaluate(real, synth, cfg, seed=0)
    assert summary.artifacts["rows_compared"] == {"real": 2500, "synth": 2500}
    assert summary.artifacts["rows_before_equalising"] == {"real": 3000, "synth": 2500}
    summary = Evaluator().evaluate(synth, real, cfg, seed=0)  # and the other way round
    assert summary.artifacts["rows_compared"] == {"real": 2500, "synth": 2500}
