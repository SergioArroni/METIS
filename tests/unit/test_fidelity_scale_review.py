"""Signs and scales of fidelity metrics checked against the paper (2026-10-09).

* R-F1: pearson, spearman and point_biserial compared |r|, so an inverted
  relation scored 1.0; the paper defines ``Delta = |T_r - T_s|`` on signed r.
* R-F2: gini_delta took the Gini of ``x - min + 1``, which depends on the unit.
* R-F3: bounded metrics that could not reach 0: delta_exceedance and
  gini_delta under ``1/(1+delta)`` (minimum 0.5) and JS in nats (minimum 0.17).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.fidelity.conditional.num_cat.point_biserial import (
    PointBiserialMetric,
)
from metis.infrastructure.metrics.fidelity.conditional.num_num.pearson import (
    PearsonCorrelationMetric,
)
from metis.infrastructure.metrics.fidelity.conditional.num_num.spearman import (
    SpearmanCorrelationMetric,
)
from metis.infrastructure.metrics.fidelity.marginal.coverage.gini_delta import GiniDeltaMetric
from metis.infrastructure.metrics.fidelity.marginal.coverage.js import JSDivergenceMetric
from metis.infrastructure.metrics.fidelity.marginal.tails.delta_exceedance import (
    DeltaExceedanceMetric,
)


def _score(metric_cls, real, synth) -> float:
    return float(metric_cls().fit(real, synth, {"seed": 0}).compute().value)


def _correlated(sign: float, n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": sign * a + 0.3 * rng.normal(size=n)})


# ── R-F1 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("metric_cls", [PearsonCorrelationMetric, SpearmanCorrelationMetric])
def test_inverted_correlation_scores_low(metric_cls):
    real, inverted = _correlated(+1.0), _correlated(-1.0, seed=1)
    assert _score(metric_cls, real, _correlated(+1.0, seed=1)) > 0.9
    assert _score(metric_cls, real, inverted) < 0.05


def _biserial(sign: float, n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b"], n)
    return pd.DataFrame({"x": sign * (g == "a") * 2.0 + rng.normal(size=n), "g": g})


def test_point_biserial_inverted_relation_scores_low():
    real = _biserial(+1.0)
    assert _score(PointBiserialMetric, real, _biserial(+1.0, seed=1)) > 0.9
    assert _score(PointBiserialMetric, real, _biserial(-1.0, seed=1)) < 0.4


def test_point_biserial_does_not_depend_on_row_order():
    real = _biserial(+1.0)
    synth = _biserial(+1.0, seed=1)
    # Put a "b" row first in the synthetic table and an "a" row first in the real one.
    real = pd.concat([real[real.g == "a"], real[real.g == "b"]], ignore_index=True)
    synth = pd.concat([synth[synth.g == "b"], synth[synth.g == "a"]], ignore_index=True)
    assert _score(PointBiserialMetric, real, synth) > 0.9


# ── R-F2 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("offset", [0.0, -50.0])
def test_gini_delta_does_not_depend_on_the_unit(offset):
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.lognormal(size=300) + offset})
    synth = pd.DataFrame({"x": rng.lognormal(sigma=0.5, size=300) + offset})
    metric = GiniDeltaMetric()
    raw = metric._compute_column(real["x"], synth["x"])
    raw_cents = metric._compute_column(real["x"] * 100, synth["x"] * 100)
    assert raw > 0.05
    assert raw_cents == pytest.approx(raw)


def test_gini_matches_the_paper_formula_on_non_negative_data():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    expected = np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) ** 2 * x.mean())
    assert GiniDeltaMetric._compute_gini(x) == pytest.approx(expected)


# ── R-F3 ─────────────────────────────────────────────────────────────────


def test_gini_delta_reaches_zero():
    n = 400
    real = pd.DataFrame({"x": np.full(n, 5.0) + np.linspace(0, 1e-6, n)})
    synth = pd.DataFrame({"x": np.r_[np.zeros(n - 1), 1000.0]})
    assert _score(GiniDeltaMetric, real, synth) < 0.01
    assert _score(GiniDeltaMetric, real, real.copy()) == pytest.approx(1.0)


def test_delta_exceedance_reaches_zero_and_one():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=1000)})
    beyond = pd.DataFrame({"x": rng.normal(size=1000) + 100.0})
    assert _score(DeltaExceedanceMetric, real, beyond) == pytest.approx(0.0, abs=1e-9)
    assert _score(DeltaExceedanceMetric, real, real.copy()) == pytest.approx(1.0)


def test_js_reaches_zero_on_disjoint_supports():
    real = pd.DataFrame({"c": ["a", "b"] * 100})
    synth = pd.DataFrame({"c": ["x", "y"] * 100})
    assert _score(JSDivergenceMetric, real, synth) == pytest.approx(0.0, abs=1e-6)
    assert _score(JSDivergenceMetric, real, real.copy()) == pytest.approx(1.0)
