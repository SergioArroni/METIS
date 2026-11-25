"""Privacy metrics must separate memorisation from an independent sample.

Audit 2026-10-08 (P1-P11): with 400 real rows, an independent sample from the
same population scored as low as an exact copy on MIA, DCR, record linkage and
k-anonymity, because the scores used absolute thresholds or looked only at the
synthetic table. These tests fix the contract of the relative metrics:

* exact copy of the training table -> score close to 0;
* independent sample of the same population -> score close to 1;
* with ``context["holdout"]`` (non-members), the control of every attack;
  without it the dataset-based metrics are not applicable (2026-10-14: the
  real->real leave-one-out substitute measured another release, so a
  generator farther from the real rows offset its copies);
* a few percent of copies is a leak: the attacks are audited in the tail
  (``audit_epsilon``), DCR and NNAA keep the average view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.privacy import (
    DCRMetric,
    InferenceAttackMetric,
    KAnonymityMetric,
    LDiversityMetric,
    MembershipInferenceMetric,
    NNAAMetric,
    RecordLinkageMetric,
    TClosenessMetric,
)
from metis.infrastructure.metrics.privacy.privacy_base import (
    MixedRecordEncoder,
    _emd,
    audit_epsilon,
    clopper_pearson,
)
from metis.infrastructure.runtime.cache import StatsStore


def _population(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    return pd.DataFrame(
        {
            "edad": np.round(40 + 12 * z[:, 0] + rng.normal(0, 4, n)),
            "ingresos": np.exp(10 + 0.5 * z[:, 0] + 0.3 * rng.normal(size=n)),
            "horas": 38 + 6 * z[:, 1] + rng.normal(0, 3, n),
            "score": 600 + 50 * z[:, 0] - 30 * z[:, 1] + rng.normal(0, 20, n),
            "hijos": rng.poisson(1.0, n),
            "sexo": np.where(rng.random(n) < 0.5, "H", "M"),
            "region": rng.choice(["N", "S", "E", "O", "C"], n),
            "y": np.where(z[:, 0] + rng.normal(0, 1, n) > 0.5, "si", "no"),
        }
    )


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    full = _population(900, seed=3)
    return {
        "real": full.iloc[:300].reset_index(drop=True),
        "holdout": full.iloc[300:600].reset_index(drop=True),
        "independent": full.iloc[600:].reset_index(drop=True),
    }


def _score(metric_cls, real, synth, context) -> float:
    result = metric_cls().fit(real, synth, dict(context)).compute()
    return float(result.value)


RELATIVE_METRICS = [
    MembershipInferenceMetric,
    DCRMetric,
    RecordLinkageMetric,
    KAnonymityMetric,
    LDiversityMetric,
    TClosenessMetric,
    InferenceAttackMetric,
]


@pytest.mark.parametrize("metric_cls", RELATIVE_METRICS, ids=lambda c: c.name)
def test_exact_copy_scores_low_with_holdout(metric_cls, tables):
    ctx = {"seed": 0, "target": "y", "holdout": tables["holdout"]}
    score = _score(metric_cls, tables["real"], tables["real"].copy(), ctx)
    assert score < 0.1, f"{metric_cls.name}: an exact copy must score ~0, got {score:.3f}"


@pytest.mark.parametrize("metric_cls", RELATIVE_METRICS, ids=lambda c: c.name)
def test_independent_sample_scores_high_with_holdout(metric_cls, tables):
    ctx = {"seed": 0, "target": "y", "holdout": tables["holdout"]}
    score = _score(metric_cls, tables["real"], tables["independent"], ctx)
    assert score > 0.85, (
        f"{metric_cls.name}: an independent sample of the population must score ~1, got {score:.3f}"
    )


@pytest.mark.parametrize("metric_cls", [*RELATIVE_METRICS, NNAAMetric], ids=lambda c: c.name)
def test_without_holdout_the_metric_is_not_applicable(metric_cls, tables):
    result = metric_cls().fit(tables["real"], tables["real"].copy(), {"seed": 0, "target": "y"})
    result = result.compute()
    assert result.status == "not_applicable"
    assert np.isnan(result.value)
    assert "holdout" in result.details["reason"]


def test_mia_without_holdout_is_nan_with_reason(tables):
    result = MembershipInferenceMetric().fit(tables["real"], tables["independent"], {}).compute()
    assert np.isnan(result.value)
    assert result.status == "not_applicable"
    assert "holdout" in result.details["reason"]


def test_partial_copy_average_views_sit_between_and_attacks_fail(tables):
    half = pd.concat(
        [tables["real"].iloc[:150], tables["independent"].iloc[:150]], ignore_index=True
    )
    ctx = {"seed": 0, "target": "y", "holdout": tables["holdout"]}
    for metric_cls in (DCRMetric, NNAAMetric):
        score = _score(metric_cls, tables["real"], half, ctx)
        assert 0.25 < score < 0.75, f"{metric_cls.name}: half copy gave {score:.3f}"
    for metric_cls in (MembershipInferenceMetric, RecordLinkageMetric):
        score = _score(metric_cls, tables["real"], half, ctx)
        assert score < 0.1, f"{metric_cls.name}: half copy is a leak, got {score:.3f}"


def _with_copies(base: pd.DataFrame, real: pd.DataFrame, fraction: float) -> pd.DataFrame:
    n = int(round(fraction * len(base)))
    return pd.concat([real.iloc[:n], base.iloc[n:]], ignore_index=True)


def test_ten_percent_of_copies_is_detected_by_the_attacks(tables):
    """30 copies among 300 synthetic rows: the averages move by ≈0.1, the
    audited attacks fall well below 1 (Fase 3 probe, 2026-10-14: 5-10 % of
    copies kept DCR and record linkage at 1.00)."""
    ctx = {"seed": 0, "target": "y", "holdout": tables["holdout"]}
    leaky = _with_copies(tables["independent"], tables["real"], 0.10)
    for metric_cls in (MembershipInferenceMetric, RecordLinkageMetric):
        score = _score(metric_cls, tables["real"], leaky, ctx)
        assert score < 0.6, f"{metric_cls.name}: 10 % of copies gave {score:.3f}"
    dcr = _score(DCRMetric, tables["real"], leaky, ctx)
    # DCR is an average: the copies move it by a few hundredths, by design.
    assert 0.8 < dcr < 0.99, dcr


def test_a_generator_far_from_the_real_rows_does_not_hide_its_copies(tables):
    """The control is the holdout against the *same* synthetic table."""
    far = tables["independent"].copy()
    for col in ("edad", "ingresos", "horas", "score"):
        far[col] = far[col] + 3 * far[col].std()
    leaky = _with_copies(far, tables["real"], 0.25)
    ctx = {"seed": 0, "target": "y", "holdout": tables["holdout"]}
    assert _score(DCRMetric, tables["real"], leaky, ctx) < 0.85
    assert _score(NNAAMetric, tables["real"], leaky, ctx) < 0.85
    assert _score(MembershipInferenceMetric, tables["real"], leaky, ctx) < 0.2
    assert _score(RecordLinkageMetric, tables["real"], leaky, ctx) < 0.2
    # Without copies the far table leaks nothing.
    for metric_cls in (DCRMetric, NNAAMetric, MembershipInferenceMetric, RecordLinkageMetric):
        assert _score(metric_cls, tables["real"], far, ctx) > 0.85, metric_cls.name


def test_audit_epsilon_is_zero_for_exchangeable_scores_and_large_for_copies():
    rng = np.random.default_rng(0)
    clean = [
        audit_epsilon(rng.normal(size=400), rng.normal(size=400))["epsilon"] for _ in range(40)
    ]
    # Bounds at 95 % confidence: an independent sample is called private in
    # at least 95 % of the draws (here all but a couple at most).
    assert np.mean(np.asarray(clean) == 0) >= 0.9
    members = rng.normal(size=400)
    members[:40] = 10.0  # 10 % of members singled out with certainty
    leak = audit_epsilon(members, rng.normal(size=400))
    assert leak["epsilon"] > 1.0 and leak["privacy_score"] < 0.4
    lo, hi = clopper_pearson(0, 400, 0.05)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 400))


def test_inference_attack_baseline_is_majority_class():
    """A 70/30 target with uninformative features must not look like a leak."""
    rng = np.random.default_rng(5)

    def sample(n):
        return pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.choice(list("abcd"), n),
                "y": np.where(rng.random(n) < 0.7, "a", "b"),
            }
        )

    ctx = {"seed": 0, "target": "y", "holdout": sample(400)}
    result = InferenceAttackMetric().fit(sample(400), sample(400), ctx)
    details = result.compute().details
    score = details["column_results"]["y"]["privacy_score"]
    # With a 1/n_classes baseline, predicting the majority class alone looked
    # like a 40 % advantage.
    assert score > 0.85, f"no-information attack on y must score ~1, got {score:.3f}"


def test_inference_attack_only_attacks_sensitive_columns(tables):
    ctx = {"seed": 0, "target": "y", "holdout": tables["holdout"]}
    result = InferenceAttackMetric().fit(tables["real"], tables["independent"], ctx).compute()
    assert list(result.details["column_results"]) == ["y"]
    custom = InferenceAttackMetric(sensitive_columns=["sexo"])
    result = custom.fit(tables["real"], tables["independent"], ctx).compute()
    assert list(result.details["column_results"]) == ["sexo"]


def test_quasi_identifiers_are_coherent_and_exclude_target(tables):
    ctx = {"seed": 0, "target": "y", "holdout": tables["holdout"]}
    qis = {}
    for metric_cls in (KAnonymityMetric, LDiversityMetric, TClosenessMetric):
        details = metric_cls().fit(tables["real"], tables["independent"], ctx).compute().details
        qis[metric_cls.name] = details["quasi_identifiers"]
        assert "y" not in details["quasi_identifiers"]
    assert qis["k_anonymity"] == qis["l_diversity"] == qis["t_closeness"]

    # Even when declared explicitly, the target is never a quasi-identifier.
    metric = KAnonymityMetric(quasi_identifiers=["sexo", "region", "y"])
    details = metric.fit(tables["real"], tables["independent"], ctx).compute().details
    assert details["quasi_identifiers"] == ["sexo", "region"]
    assert details["excluded_from_quasi_identifiers"] == ["y"]


def test_t_closeness_reads_target_from_context(tables):
    ctx = {"seed": 0, "target": "y", "holdout": tables["holdout"]}
    details = TClosenessMetric().fit(tables["real"], tables["independent"], ctx).compute().details
    assert details["sensitive_columns"] == ["y"]
    assert list(details["per_column"]) == ["y"]


def test_mixed_encoder_uses_categoricals_and_missingness():
    real = pd.DataFrame({"x": [0.0, 1.0, np.nan, 3.0], "c": ["a", "b", "a", "b"]})
    enc = MixedRecordEncoder(real, ["x", "c"])
    a = enc.transform(pd.DataFrame({"x": [1.0], "c": ["a"]}))
    b = enc.transform(pd.DataFrame({"x": [1.0], "c": ["b"]}))
    # A categorical mismatch adds exactly 1 to the squared distance.
    assert np.isclose(((a - b) ** 2).sum(), 1.0)
    # Missingness is encoded, not silently mapped to the raw value 0.
    missing = enc.transform(pd.DataFrame({"x": [np.nan], "c": ["a"]}))
    zero = enc.transform(pd.DataFrame({"x": [0.0], "c": ["a"]}))
    assert not np.allclose(missing, zero)


def test_distance_space_is_shared_through_stats_store(tables):
    store = StatsStore()
    ctx = {"seed": 0, "target": "y", "holdout": tables["holdout"], "stats_store": store}
    for metric_cls in (DCRMetric, NNAAMetric, MembershipInferenceMetric):
        metric_cls().fit(tables["real"], tables["independent"], ctx).compute()
    keys = [k for k in store._cache if k.startswith("privacy:distance_space:")]
    assert len(keys) == 1, keys


def test_t_closeness_emd_reproduces_li_et_al_salary_example():
    """Li et al. (2007): salaries 3k..11k uniform in the table; the class
    {3k, 4k, 5k} is at EMD 0.375 and {6k, 8k, 11k} at 0.167 (ordered
    distance), while the total variation puts both at 2/3."""
    table = np.full(9, 1 / 9)
    classes = np.zeros((2, 9))
    classes[0, [0, 1, 2]] = 1 / 3
    classes[1, [3, 5, 8]] = 1 / 3
    order = np.arange(9, dtype=float)
    assert _emd(classes, table, order) == pytest.approx([0.375, 1 / 6])
    # Missing values (outside the order) are at distance 1: only their mass counts.
    with_missing = np.append(order, np.nan)
    table_m = np.append(table * 0.5, 0.5)
    only_missing = np.zeros((1, 10))
    only_missing[0, 9] = 1.0
    assert _emd(only_missing, table_m, with_missing) == pytest.approx([0.5])
    assert _emd(np.append(classes, [[0], [0]], axis=1), np.append(table, 0.0), with_missing) == (
        pytest.approx([0.375, 1 / 6])
    )


def test_t_closeness_orders_a_numeric_sensitive_column():
    """A class that only moves the sensitive value to the next bin is closer
    to the table than one that moves it to the far end (equal for the TV)."""
    rng = np.random.default_rng(0)
    n = 600
    real = pd.DataFrame({"qi": rng.choice(list("ab"), n), "s": rng.integers(0, 5, n).astype(float)})
    near, far = real.copy(), real.copy()
    near.loc[near["qi"] == "a", "s"] = 1.0
    far.loc[far["qi"] == "a", "s"] = 4.0
    near.loc[near["qi"] == "b", "s"] = 0.0
    far.loc[far["qi"] == "b", "s"] = 0.0
    out = {}
    for name, synth in (("near", near), ("far", far)):
        res = (
            TClosenessMetric(sensitive_column="s", quasi_identifiers=["qi"])
            .fit(real, synth, {"seed": 0, "holdout": real.sample(200, random_state=1)})
            .compute()
        )
        out[name] = res.details["per_column"]["s"]["median_t_matched_classes"]
        assert res.details["ground_distance"] == {"s": "ordered"}
    assert out["near"] < out["far"]
