"""Missing values are kept: every metric uses what it has (decision 2026-10-09, option 3).

The preprocessing used to drop every row with a NaN in any column, from the
real, synthetic and holdout tables, before every metric: in Hefesto's prior
METIS saw 55 % of the rows on average and the synthetic gaps were invisible.
Unreadable values become gaps and are counted; the rules axis checks them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from metis.infrastructure.metrics.fidelity.fidelity_base import standardise_with_missingness
from metis.interface.sdk import Evaluator

SCHEMA = {
    "x": "continuous",
    "z": "continuous",
    "c": "nominal",
    "o": {"type": "ordinal", "levels": ["lo", "mid", "hi"]},
    "y": "nominal",
}


def _table(n: int, seed: int, gaps: float = 0.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "z": x + rng.normal(size=n),
            "c": rng.choice(list("abc"), n),
            "o": rng.choice(["lo", "mid", "hi"], n),
            "y": np.where(x + rng.normal(size=n) > 0, "s", "n"),
        }
    )
    for col in ("x", "z", "c", "o"):
        df.loc[rng.random(n) < gaps, col] = np.nan
    return df


def _evaluate(real, synth, metrics, n_splits=2, **kwargs):
    cfg = {
        "data": {"schema": SCHEMA, "target": "y"},
        "metrics": metrics,
        "evaluation": {"n_runs": 1, "ml_efficiency_config": {"n_splits": n_splits}},
        "reproducibility": {"seed": 0},
    }
    return Evaluator().evaluate(real, synth, cfg, **kwargs)


def _result(summary, metric_id):
    (r,) = [r for r in summary.results if r.id == metric_id]
    return r


def test_no_row_is_lost_because_of_a_gap():
    real, synth = _table(300, 0), _table(300, 1)
    assert real.isna().any(axis=1).mean() > 0.5  # most rows have a gap
    summary = _evaluate(real, synth, ["fidelity.ks"], holdout=_table(100, 2))
    rows = summary.artifacts["rows"]
    for label, n in (("real", 300), ("synth", 300), ("holdout", 100)):
        assert rows[label]["rows_in"] == rows[label]["rows_out"] == n
    assert rows["real"]["rows_with_missing"] == int(real.isna().any(axis=1).sum())
    assert summary.artifacts["synth_rows_dropped"] == 0


def test_marginal_metric_uses_the_observed_values_of_each_column():
    real, synth = _table(300, 0), _table(300, 1)
    ks = _result(_evaluate(real, synth, ["fidelity.ks"]), "fidelity.ks")
    by_hand = stats.ks_2samp(real["x"].dropna(), synth["x"].dropna()).statistic
    assert ks.details["x"]["raw_value"] == pytest.approx(by_hand)


def test_pair_metric_uses_the_rows_where_both_columns_are_observed():
    real, synth = _table(300, 0), _table(300, 1)
    res = _result(_evaluate(real, synth, ["fidelity.pearson"]), "fidelity.pearson")
    both = real[["x", "z"]].dropna()
    by_hand = stats.pearsonr(both["x"], both["z"])[0]
    assert res.details["x|z"]["real_value"] == pytest.approx(by_hand)


def test_column_missing_in_the_whole_real_table_does_not_apply():
    real, synth = _table(200, 0), _table(200, 1)
    real["x"] = np.nan
    real["z"] = np.nan
    real["o"] = np.nan  # its numeric view ``o__ord`` is a numeric column too
    res = _result(_evaluate(real, synth, ["fidelity.wasserstein"]), "fidelity.wasserstein")
    assert res.status == "not_applicable"


def test_unreadable_value_becomes_a_gap_in_every_view_and_is_counted():
    real, synth = _table(200, 0, gaps=0.0), _table(200, 1, gaps=0.0)
    synth.loc[[0, 1, 2, 3], "o"] = "huge"  # not a level of the ordinal
    summary = _evaluate(real, synth, ["fidelity.tvd"])
    rows = summary.artifacts["rows"]["synth"]
    assert rows["cells_invalid"] == 4 and rows["invalid_by_column"] == {"o": 4}
    assert rows["rows_out"] == 200


def test_utility_keeps_gaps_in_the_features_and_drops_rows_without_target():
    # Big enough for the real signal to clear 3× its sampling noise.
    real, synth = _table(2000, 0), _table(2000, 1)
    real.loc[:9, "y"] = np.nan
    res = _result(
        _evaluate(real, synth, ["utility.ml_efficiency"], n_splits=5, holdout=_table(2000, 2)),
        "utility.ml_efficiency",
    )
    assert res.status == "ok", res.details.get("error")
    assert res.details["rows_without_target"] == {"real": 10}


def test_privacy_metrics_accept_gaps():
    real, synth = _table(300, 0), _table(300, 1)
    summary = _evaluate(
        real,
        synth,
        [
            "privacy.dcr",
            "privacy.nnaa",
            "privacy.mia",
            "privacy.record_linkage",
            "privacy.k_anonymity",
            "privacy.inference_attack",
        ],
        holdout=_table(200, 2),
    )
    assert {r.status for r in summary.results} == {"ok"}, {
        r.id: r.details.get("error") for r in summary.results
    }


def test_joint_standardisation_without_gaps_is_the_former_one():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(50, 3)), rng.normal(size=(40, 3))
    Xs, Ys, k = standardise_with_missingness(X, Y)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-10
    np.testing.assert_allclose(Xs, (X - mean) / std)
    np.testing.assert_allclose(Ys, (Y - mean) / std)
    assert k == 0
    Y[0, 1] = np.nan
    Xs, Ys, k = standardise_with_missingness(X, Y)
    assert k == 1 and Ys.shape[1] == 4 and Ys[0, 1] == 0.0 and Ys[0, 3] == 1.0
