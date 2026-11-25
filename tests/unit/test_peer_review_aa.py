"""Findings reported from Hefesto's SPIKE-0072/0076 (48,000 scored pairs).

* not-applicable metrics were indistinguishable from failed ones;
* privacy without a holdout gave a silent NaN for MIA;
* fidelity ids were short ('ks') while utility/privacy were 'family.id';
* ``composite_score`` vanished when no family had a score;
* every METIS reported version 0.1.0;
* NNAA broke distance ties towards the real rows: with discrete data an
  independent sample scored 0 (as a copy).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from metis.application.aggregator import Aggregator
from metis.domain.entities import MetricResult
from metis.infrastructure.metrics.privacy.dataset_based.empirical_similarity.nnaa import NNAAMetric
from metis.infrastructure.metrics.privacy.privacy_base import MissingHoldoutWarning
from metis.interface.sdk import Evaluator


def _categorical(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.choice(list("xyz"), n),
            "b": rng.choice(list("uv"), n),
            "c": rng.choice(list("pq"), n),
        }
    )


_SCHEMA = {"a": "nominal", "b": "nominal", "c": "nominal"}


def _evaluate(metrics, **kwargs):
    cfg = {
        "data": {"schema": _SCHEMA},
        "metrics": metrics,
        "evaluation": {"n_runs": 1},
        "reproducibility": {"seed": 0},
    }
    return Evaluator().evaluate(_categorical(200, 0), _categorical(200, 1), cfg, **kwargs)


def test_status_is_derived_from_value_and_details():
    ok = MetricResult(id="fidelity.ks", value=0.5, details={}, family="fidelity")
    err = MetricResult(
        id="fidelity.ks", value=float("nan"), details={"error": "x"}, family="fidelity"
    )
    na = MetricResult.not_applicable("fidelity.ks", "fidelity", "no numeric column")
    assert (ok.status, err.status, na.status) == ("ok", "error", "not_applicable")
    assert na.details["reason"] == "no numeric column"
    with pytest.raises(ValueError, match="status"):
        MetricResult(id="fidelity.ks", value=1.0, details={}, family="fidelity", status="meh")


def test_not_applicable_is_reported_apart_from_failures():
    summary = _evaluate(["fidelity.marginal.tails"])
    agg = summary.aggregates
    assert {r.status for r in summary.results} == {"not_applicable"}
    assert "fidelity.ks" in agg["fidelity_not_applicable"]
    assert "fidelity_failed_metrics" not in agg
    assert agg["failed_metrics"] == 0
    assert agg["not_applicable_metrics"] == len(summary.results)


def test_all_result_ids_carry_their_family():
    summary = _evaluate(["fidelity.marginal.coverage", "privacy.dcr"], holdout=_categorical(80, 2))
    assert summary.results
    for r in summary.results:
        assert r.id.split(".", 1)[0] == r.family


def test_composite_score_is_always_present():
    agg = Aggregator().aggregate(
        [MetricResult.not_applicable("fidelity.ks", "fidelity", "no numeric column")], None
    )
    assert math.isnan(agg["composite_score"])
    assert agg["composite_families"] == []


def test_privacy_without_holdout_warns():
    with pytest.warns(MissingHoldoutWarning, match="holdout"):
        summary = _evaluate(["privacy.mia", "privacy.dcr"])
    (mia,) = [r for r in summary.results if r.id == "privacy.mia"]
    assert mia.status == "not_applicable"
    assert any("holdout" in w for w in summary.artifacts["warnings"])


def test_run_records_the_metis_version_and_commit():
    summary = _evaluate(["fidelity.marginal.coverage"])
    assert summary.artifacts["metis_version"]
    assert "metis_commit" in summary.artifacts


def test_nnaa_splits_ties_instead_of_favouring_real_rows():
    real, independent = _categorical(300, 0), _categorical(300, 1)
    ctx = {"seed": 0, "holdout": _categorical(300, 2)}
    independent_score = NNAAMetric().fit(real, independent, ctx).compute()
    copy_score = NNAAMetric().fit(real, real.copy(), ctx).compute()
    # Only 12 distinct records: every distance is a tie.  Before, the first
    # index (always a real row) won every tie and the independent sample scored 0.
    assert independent_score.value > 0.9
    assert independent_score.details["excess_risk_signed"] < 0.2
    # A copy still has every real record at distance 0 and scores no better.
    assert copy_score.value <= independent_score.value


def test_rows_with_gaps_are_kept_and_single_class_target_does_not_apply():
    real = _categorical(200, 0)
    real.loc[real.index[:150], "b"] = None  # only 50 complete rows: all 200 are kept
    real["y"] = "same"  # one class
    synth = _categorical(200, 1)
    synth["y"] = "same"
    cfg = {
        "data": {"schema": {**_SCHEMA, "y": "nominal"}, "target": "y"},
        "metrics": ["utility.ml_efficiency"],
        "evaluation": {"n_runs": 1},
        "reproducibility": {"seed": 0},
    }
    summary = Evaluator().evaluate(real, synth, cfg)
    assert summary.artifacts["rows"]["real"]["rows_out"] == 200
    assert summary.artifacts["rows"]["real"]["rows_with_missing"] == 150
    (util,) = summary.results
    assert util.status == "not_applicable"
