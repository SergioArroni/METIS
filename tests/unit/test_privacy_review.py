"""Second review of the privacy metrics (2026-10-09).

Each test pins a defect found after the audit fixes:

* R-P1: an incomplete holdout was dropped without notice.
* R-P2: NNAA compared with a fixed 0.5 whatever the size ratio.
* R-P3: unseen categories were closer to every real category than a mismatch.
* R-P4: DCR's SDMetrics share mixed searched tables of different sizes and
  compared distances across records instead of per record.
* R-P5: NNAA with ``exclude_self=False`` always scored 1.
* R-P6/R-P7: differential privacy scored an invented epsilon, NaN/negative
  epsilons as perfect privacy, and let the model override a declared budget.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.privacy import (
    DCRMetric,
    DifferentialPrivacyMetric,
    InferenceAttackMetric,
    LDiversityMetric,
    MembershipInferenceMetric,
    NNAAMetric,
)
from metis.infrastructure.metrics.privacy.privacy_base import (
    HoldoutColumnsWarning,
    MixedRecordEncoder,
)


def _population(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    return pd.DataFrame(
        {
            "edad": np.round(40 + 12 * z[:, 0] + rng.normal(0, 4, n)),
            "horas": 38 + 6 * z[:, 1] + rng.normal(0, 3, n),
            "score": 600 + 50 * z[:, 0] - 30 * z[:, 1] + rng.normal(0, 20, n),
            "sexo": np.where(rng.random(n) < 0.5, "H", "M"),
            "region": rng.choice(["N", "S", "E", "O", "C"], n),
            "y": np.where(z[:, 0] + rng.normal(0, 1, n) > 0.5, "si", "no"),
        }
    )


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    full = _population(1500, seed=7)
    return {
        "real": full.iloc[:300].reset_index(drop=True),
        "holdout": full.iloc[300:600].reset_index(drop=True),
        "independent": full.iloc[600:].reset_index(drop=True),
    }


def _ctx(**extra) -> dict:
    return {"seed": 0, "target": "y", **extra}


# ── R-P1: incomplete holdout ─────────────────────────────────────────────


def test_holdout_missing_a_column_warns_and_is_used_over_the_rest(tables):
    holdout = tables["holdout"].drop(columns=["region"])
    with pytest.warns(HoldoutColumnsWarning, match="region"):
        result = (
            DCRMetric().fit(tables["real"], tables["real"].copy(), _ctx(holdout=holdout)).compute()
        )
    assert result.details["n_holdout"] == len(holdout)
    assert "region" not in result.details["columns"]
    assert result.details["holdout"] == {
        "used": True,
        "missing_columns": ["region"],
        "ignored_reason": None,
    }
    with pytest.warns(HoldoutColumnsWarning):
        mia = (
            MembershipInferenceMetric()
            .fit(tables["real"], tables["real"].copy(), _ctx(holdout=holdout))
            .compute()
        )
    assert mia.value < 0.1  # a copy is caught: the holdout was not discarded


def test_holdout_without_the_sensitive_column_is_not_its_control(tables):
    holdout = tables["holdout"].drop(columns=["y"])
    with pytest.warns(HoldoutColumnsWarning, match="sensitive"):
        result = (
            LDiversityMetric()
            .fit(tables["real"], tables["independent"], _ctx(holdout=holdout))
            .compute()
        )
    # No control for the sensitive column: an error with the reason, not a
    # silent switch to the real->real reference (2026-10-14).
    assert np.isnan(result.value)
    assert result.status == "error"
    assert "y" in result.details["error"]
    with pytest.warns(HoldoutColumnsWarning):
        attack = (
            InferenceAttackMetric()
            .fit(tables["real"], tables["independent"], _ctx(holdout=holdout))
            .compute()
        )
    assert attack.status == "error" and "y" in attack.details["error"]


def test_unusable_holdout_warns_and_is_reported(tables):
    with pytest.warns(HoldoutColumnsWarning, match="empty"):
        result = (
            MembershipInferenceMetric()
            .fit(tables["real"], tables["independent"], _ctx(holdout=tables["holdout"].iloc[:0]))
            .compute()
        )
    assert np.isnan(result.value)
    assert "empty" in result.details["error"]
    assert result.details["holdout"]["used"] is False


def test_complete_holdout_does_not_warn(tables):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HoldoutColumnsWarning)
        result = (
            DCRMetric()
            .fit(tables["real"], tables["independent"], _ctx(holdout=tables["holdout"]))
            .compute()
        )
    assert result.details["holdout"]["missing_columns"] == []


# ── R-P2 / R-P5: NNAA ────────────────────────────────────────────────────


@pytest.mark.parametrize("n_synth", [100, 300, 900])
def test_nnaa_independent_sample_scores_high_for_any_size_ratio(tables, n_synth):
    synth = tables["independent"].iloc[:n_synth].reset_index(drop=True)
    ctx = _ctx(holdout=tables["holdout"])
    result = NNAAMetric().fit(tables["real"], synth, ctx).compute()
    assert result.value > 0.85, result.details["privacy_loss"]


def test_nnaa_copy_still_scores_low(tables):
    ctx = _ctx(holdout=tables["holdout"])
    result = NNAAMetric().fit(tables["real"], tables["real"].copy(), ctx).compute()
    assert result.value < 0.1


def test_nnaa_refuses_exclude_self_false():
    with pytest.raises(ValueError, match="exclude_self"):
        NNAAMetric(exclude_self=False)


# ── R-P3: unseen categories ──────────────────────────────────────────────


def test_unseen_category_is_as_far_as_a_seen_mismatch():
    enc = MixedRecordEncoder(pd.DataFrame({"c": ["a", "b"]}), ["c"])
    a, b, unseen = enc.transform(pd.DataFrame({"c": ["a", "b", "zzz"]}))
    assert np.isclose(((a - b) ** 2).sum(), 1.0)
    assert np.isclose(((a - unseen) ** 2).sum(), 1.0)


# ── R-P4: DCR share against the holdout ──────────────────────────────────


def test_dcr_share_matches_its_expectation_for_an_independent_sample_and_small_holdout(
    tables,
):
    result = (
        DCRMetric()
        .fit(
            tables["real"],
            tables["independent"].iloc[:300],
            _ctx(holdout=tables["holdout"].iloc[:60]),
        )
        .compute()
    )
    # Training and holdout pooled: the expected share is n_train / (n_train + n_holdout).
    expected = len(tables["real"]) / (len(tables["real"]) + 60)
    assert result.details["share_expected"] == pytest.approx(expected)
    assert abs(result.details["share_closest_is_training"] - expected) < 0.1
    assert result.value > 0.85


# ── R-P6 / R-P7: differential privacy ────────────────────────────────────


class _Model:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _dp(metric, model=None):
    ctx = {"seed": 0} if model is None else {"seed": 0, "generative_model": model}
    data = pd.DataFrame({"x": [1.0, 2.0]})
    return metric.fit(data, data, ctx).compute()


def test_dp_without_any_epsilon_is_nan_not_an_invented_score():
    result = _dp(DifferentialPrivacyMetric())
    assert np.isnan(result.value)
    assert result.status == "not_applicable"
    assert "epsilon" in result.details["reason"]


def test_dp_declared_epsilon_wins_over_the_model():
    result = _dp(DifferentialPrivacyMetric(epsilon=1.0), _Model(epsilon=8.0))
    assert result.details["epsilon"] == 1.0
    assert result.details["epsilon_source"] == "configured"
    assert np.isclose(result.value, np.exp(-0.5))


def test_dp_model_epsilon_is_used_when_none_is_declared():
    result = _dp(DifferentialPrivacyMetric(), _Model(epsilon=2.0))
    assert result.details["epsilon_source"] == "model_attribute"
    assert np.isclose(result.value, np.exp(-1.0))


@pytest.mark.parametrize("eps", [float("nan"), -1.0])
def test_dp_invalid_epsilon_is_an_error_not_perfect_privacy(eps):
    result = _dp(DifferentialPrivacyMetric(), _Model(epsilon=eps))
    assert np.isnan(result.value)
    assert "Invalid epsilon" in result.details["error"]


# ── R-P8: ordinals count once ────────────────────────────────────────────


def _with_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Mimic the caster: an ordinal keeps ``<name>`` plus ``<name>__ord``."""
    out = df.copy()
    levels = {"N": 0.0, "S": 0.25, "E": 0.5, "O": 0.75, "C": 1.0}
    out["region__ord"] = out["region"].map(levels)
    out["y__ord"] = (out["y"] == "si").astype(float)
    return out


def test_ordinal_shadows_do_not_enter_as_separate_columns(tables):
    real = _with_ordinals(tables["real"])
    synth = _with_ordinals(tables["independent"])
    ctx = _ctx(holdout=_with_ordinals(tables["holdout"]))
    dcr = DCRMetric().fit(real, synth, dict(ctx)).compute()
    assert "region__ord" not in dcr.details["columns"]
    assert "region" in dcr.details["columns"]
    ldiv = LDiversityMetric().fit(real, synth, dict(ctx)).compute()
    # The target's shadow used to be a quasi-identifier: the target itself.
    assert "y__ord" not in ldiv.details["quasi_identifiers"]
    assert "region__ord" not in ldiv.details["quasi_identifiers"]


# ── R-P9: vocabulary of near-unique columns (2026-10-18) ────────────────


def test_categories_absent_from_the_real_table_are_still_told_apart():
    real = pd.DataFrame({"c": ["a", "b"]})
    other = pd.DataFrame({"c": ["x", "y"]})
    enc = MixedRecordEncoder(real, ["c"], vocabulary_from=[other])
    a, x, y = enc.transform(pd.DataFrame({"c": ["a", "x", "y"]}))
    assert np.isclose(((x - y) ** 2).sum(), 1.0)  # before: both in the unseen slot, at 0
    assert np.isclose(((a - x) ** 2).sum(), 1.0)
    assert np.isclose(((x - x) ** 2).sum(), 0.0)


def _with_name(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # A near-unique text column, as airbnb ``name``.
    return frame.assign(nombre=[f"{prefix}{i}" for i in range(len(frame))])


def test_a_near_unique_column_does_not_pull_the_synthetic_table_to_the_holdout(tables):
    real = _with_name(tables["real"], "r")
    holdout = _with_name(tables["holdout"], "h")
    independent = _with_name(tables["independent"].iloc[:300], "s")
    clean = DCRMetric().fit(real, independent, _ctx(holdout=holdout)).compute()
    # An independent sample: the share stays at its expectation (it was ~0 with
    # the real-only vocabulary: every holdout and synthetic name in one slot).
    assert abs(clean.details["share_closest_is_training"] - 0.5) < 0.1
    copies = pd.concat([independent.iloc[:240], real.iloc[:60]], ignore_index=True)
    leaky = DCRMetric().fit(real, copies, _ctx(holdout=holdout)).compute()
    assert leaky.details["share_closest_is_training"] > clean.details["share_closest_is_training"]
