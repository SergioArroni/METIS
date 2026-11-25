"""One representation of the ordinals for all of METIS (decision 2026-10-09).

An ordinal has two views: its level (categorical) and its rank ``i/(K-1)``
(``<name>__ord``).  What measures category frequencies uses the level; what
measures distance or order uses the rank; models get both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.fidelity.conditional.num_cat.eta_squared import EtaSquaredMetric
from metis.infrastructure.metrics.utility.ml_efficiency import catboost_trainer as ct
from metis.infrastructure.metrics.utility.ml_efficiency_metric import _no_signal_score
from metis.interface.sdk import Evaluator

LEVELS = ["l0", "l1", "l2", "l3", "l4"]


def _table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    df = pd.DataFrame({"x": x, **{f"x{k}": rng.standard_normal(n) for k in range(2, 8)}})
    for j in range(3):  # ordinals driven by x, so they carry signal
        idx = np.clip(np.round(2 + 1.2 * x + rng.standard_normal(n) * 0.8), 0, 4).astype(int)
        df[f"o{j}"] = [LEVELS[i] for i in idx]
    df["c"] = rng.choice(["a", "b", "c"], n)
    return df


def _schema(target: str | None = None) -> dict:
    schema = {"x": "continuous", "c": "nominal"}
    schema.update({f"x{k}": "continuous" for k in range(2, 8)})
    schema.update({f"o{j}": {"type": "ordinal", "levels": LEVELS} for j in range(3)})
    data = {"schema": schema}
    if target:
        data["target"] = target
    return data


def _move_one_level(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for j in range(3):
        idx = out[f"o{j}"].map(LEVELS.index).to_numpy()
        step = rng.choice([-1, 1], len(idx))
        idx = np.where((idx + step < 0) | (idx + step > 4), idx - step, idx + step)
        out[f"o{j}"] = [LEVELS[i] for i in idx]
    return out


def _evaluate(real, synth, metrics, holdout=None, target=None):
    cfg = {
        "data": _schema(target),
        "metrics": metrics,
        "evaluation": {"n_runs": 1, "ml_efficiency_config": {"n_splits": 3}},
        "reproducibility": {"seed": 0},
    }
    return Evaluator().evaluate(real, synth, cfg, holdout=holdout)


# ── privacy: distances use the rank, unscaled ──────────────────────────────


def test_near_copy_with_ordinals_moved_one_level_is_caught_by_distances():
    real, holdout = _table(300, 0), _table(300, 1)
    near_copy = _move_one_level(real, 2)
    num = ["x"] + [f"x{k}" for k in range(2, 8)]
    near_copy[num] += np.random.default_rng(3).normal(0, 0.05, (300, len(num)))
    summary = _evaluate(real, near_copy, ["privacy.dcr", "privacy.nnaa"], holdout=holdout)
    scores = {r.id: r.value for r in summary.results}
    # As categories one level away cost a full mismatch per ordinal and the
    # copy looked private; through the rank it is a near-copy.
    assert scores["privacy.dcr"] < 0.3


def test_independent_sample_raises_no_false_alarm():
    real, holdout, independent = _table(300, 0), _table(300, 1), _table(300, 4)
    summary = _evaluate(real, independent, ["privacy.dcr", "privacy.nnaa"], holdout=holdout)
    assert all(r.value > 0.85 for r in summary.results)


# ── fidelity: each pair of variables once per metric ───────────────────────


def test_rank_is_paired_with_nominals_and_never_with_its_own_level():
    df = _table(200, 0)
    df["o0__ord"] = df["o0"].map(LEVELS.index) / 4.0
    pairs = EtaSquaredMetric().fit(df, df, {})._pairs
    assert ("o0__ord", "c") in pairs
    assert ("o0__ord", "o0") not in pairs
    assert ("x", "o0") in pairs  # numeric vs ordinal: through the level
    variables = [tuple(sorted(p.replace("__ord", "") for p in pair)) for pair in pairs]
    assert len(variables) == len(set(variables))  # no pair of variables twice


# ── utility: both views for predictors, QWK for an ordinal target ─────────


def test_quadratic_weighted_kappa_grades_the_distance_between_levels():
    t = np.array([0, 0.25, 0.5, 0.75, 1.0] * 20)
    assert ct.quadratic_weighted_kappa(t, t) == pytest.approx(1.0)
    one_off = ct.quadratic_weighted_kappa(t, np.clip(t + 0.25, 0, 1))
    far_off = ct.quadratic_weighted_kappa(t, 1.0 - t)
    assert 0 < far_off < one_off < 1 or far_off < 0 < one_off < 1
    assert ct.quadratic_weighted_kappa(t, np.full_like(t, 0.5)) == pytest.approx(0.0)
    assert _no_signal_score(t, t, "ordinal") == 0.0


def test_ordinal_target_is_scored_with_qwk_and_a_shift_costs_less_than_noise():
    real, holdout = _table(400, 0), _table(400, 1)
    good = _table(400, 5)
    shifted = good.copy()
    shifted["o0"] = [LEVELS[min(LEVELS.index(v) + 1, 4)] for v in shifted["o0"]]
    noise = good.copy()
    noise["o0"] = np.random.default_rng(6).choice(LEVELS, 400)
    scores = {}
    for name, synth in (("good", good), ("shifted", shifted), ("noise", noise)):
        (r,) = _evaluate(real, synth, ["utility.ml_efficiency"], holdout, target="o0").results
        assert r.details["task_type"] == "ordinal", r.details
        assert r.details["metric"].startswith("QWK")
        scores[name] = r.value
    assert scores["good"] > scores["shifted"] > scores["noise"]
