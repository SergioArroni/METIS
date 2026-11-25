"""Method of the utility metrics (audit findings U2-U7).

* U2: early stopping never sees the test rows.
* U3: absolute score between a model without signal and TTR, TSTR by
  default, repeated splits.
* U4: no hyperparameter search: CatBoost defaults with early stopping.
* U5: float targets are mapped, not truncated.
* U6: a failing strategy is reported, never silently dropped.
* U7: classification/regression efficiency share the ml_efficiency code.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.utility import ml_efficiency_metric as mem
from metis.infrastructure.metrics.utility.ml_common import ML_CONFIG_KEYS
from metis.infrastructure.metrics.utility.ml_efficiency import catboost_trainer as ct
from metis.infrastructure.metrics.utility.ml_efficiency.base import StrategyResult, TrainingStrategy
from metis.infrastructure.metrics.utility.ml_efficiency.classification_efficiency import (
    ClassificationEfficiencyMetric,
)
from metis.infrastructure.metrics.utility.ml_efficiency.regression_efficiency import (
    RegressionEfficiencyMetric,
)


@pytest.fixture(autouse=True)
def _clean_ttr_cache():
    mem.clear_ttr_cache()
    yield
    mem.clear_ttr_cache()


class _Spec:
    def __init__(self, target):
        self.target_list = [target]
        self.task_type = None
        self.schema = {}


def _table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    c = rng.choice(["a", "b"], size=n)
    y = np.where(x1 + (c == "a") > 0.5, "yes", "no")
    return pd.DataFrame({"x1": x1, "c": c, "y": y})


def _context(ml_config: dict | None = None, **extra) -> dict:
    ctx = {
        "seed": 0,
        "dataset_spec": _Spec("y"),
        "config": {"evaluation": {"ml_efficiency_config": ml_config or {}}},
        # Utility needs a holdout (real rows the generator never saw) since 2026-10-19.
        "holdout": _table(30, 77),
    }
    ctx.update(extra)
    return ctx


def _result(strategy, scores, valid=True, error=None) -> StrategyResult:
    return StrategyResult(
        strategy=strategy,
        score=float(np.mean(scores)) if scores else float("nan"),
        raw_metric=0.0,
        scores_per_run=list(scores),
        std=0.0,
        best_params={},
        is_valid=valid,
        error=error,
    )


def _fake_strategies(monkeypatch, scores: dict[str, list[float] | None], chance: float = 0.5):
    """Replace model training by fixed per-split scores (``None`` = failure).

    The model without signal scores *chance* on every split.  ``"half"``
    gives the TTR scores on half-samples (default: none, so the noise falls
    back to the spread of the TTR splits).
    """
    seen: list[TrainingStrategy] = []
    monkeypatch.setattr(mem, "_chance_scores", lambda *a, **k: [chance] * k["n_splits"])
    monkeypatch.setattr(
        mem,
        "_half_sample_scores",
        lambda *a, **k: _result(TrainingStrategy.TTR, scores.get("half") or []),
    )

    def fake(strategy, *args, **kwargs):
        seen.append(strategy)
        s = scores[strategy.value]
        if s is None:
            return _result(strategy, [], valid=False, error="boom")
        return _result(strategy, s)

    monkeypatch.setattr(mem, "_evaluate_strategy", fake)
    return seen


# ── U2 ───────────────────────────────────────────────────────────────────


class _Recorder:
    predicted: list[np.ndarray] = []

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, **kwargs):
        self._label = y[0]
        self.classes_ = np.unique(y)
        if "eval_set" in kwargs:
            _Recorder.validated.append(np.asarray(kwargs["eval_set"][0]["a"]))
        return self

    def predict(self, X):
        _Recorder.predicted.append(np.asarray(X["a"]))
        return np.repeat(self._label, len(X))

    def predict_proba(self, X):
        # The AUC scores probabilities: the rows scored are recorded here.
        _Recorder.predicted.append(np.asarray(X["a"]))
        return np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))


def test_early_stopping_never_sees_the_test_rows(monkeypatch):
    _Recorder.predicted, _Recorder.validated = [], []
    monkeypatch.setenv("METIS_N_JOBS", "1")
    monkeypatch.setattr(ct.ClassificationTrainer, "_get_model_class", lambda self: _Recorder)
    trainer = ct.ClassificationTrainer(n_runs=1)
    X = pd.DataFrame({"a": np.arange(50.0)})
    y = np.array([0, 1] * 25)
    trainer.train_and_evaluate(X[:40], X[40:].reset_index(drop=True), y[:40], y[40:], seed=0)

    (rows,) = _Recorder.validated  # validation slice of the training rows only
    assert rows.max() < 40
    assert len(rows) == 8  # 20 % of 40
    (final,) = _Recorder.predicted
    assert set(final) == set(np.arange(40.0, 50.0))


def test_final_fit_uses_the_early_stopped_tree_count(monkeypatch):
    """And the learning rate CatBoost chose for it: with fewer trees it would pick another."""
    _Recorder.validated, _Recorder.predicted = [], []

    class _Stopping(_Recorder):
        def get_best_iteration(self):
            return 6

        def get_all_params(self):
            return {"learning_rate": 0.07}

    monkeypatch.setattr(ct.ClassificationTrainer, "_get_model_class", lambda self: _Stopping)
    trainer = ct.ClassificationTrainer(n_runs=1)
    X = pd.DataFrame({"a": np.arange(30.0)})
    y = np.array([0, 1] * 15)
    best, *_ = trainer.train_and_evaluate(X[:20], X[20:], y[:20], y[20:], seed=0)
    assert best["iterations"] == 7
    assert best["learning_rate"] == pytest.approx(0.07)


def test_each_split_early_stops_on_its_own_training_rows(monkeypatch):
    """Nothing chosen on the first split is reused on the others (the old leak)."""
    seeds: list[int] = []
    original = ct.CatBoostTrainer.tune

    def spy(self, X, y, seed=42, cat_features=None):
        seeds.append(seed)
        return original(self, X, y, seed, cat_features)

    monkeypatch.setattr(ct.CatBoostTrainer, "tune", spy)
    ctx = _context({"n_splits": 3})
    mem.MLEfficiencyMetric().fit(_table(60, 0), _table(60, 1), ctx).compute()
    # Every fitted model (TTR, its half-sample and, with signal, TSTR) tunes
    # once per split on that split's own rows.
    counts = [seeds.count(k) for k in range(3)]
    assert len(seeds) == sum(counts) and counts[0] >= 2 and len(set(counts)) == 1


# ── U4 ───────────────────────────────────────────────────────────────────


def test_no_search_catboost_defaults():
    params = ct.ClassificationTrainer(n_runs=1)._base_params(0)
    assert params["iterations"] == ct.MAX_ITERATIONS == 1000  # CatBoost's default
    for searched in ("depth", "learning_rate", "l2_leaf_reg", "bagging_temperature"):
        assert searched not in params


def test_catboost_runs_are_deterministic():
    X = _table(120, 0)
    y = X.pop("y").to_numpy()
    runs = [
        ct.ClassificationTrainer(n_runs=1, thread_count=t).train_and_evaluate(
            X[:90], X[90:], y[:90], y[90:], seed=3
        )[1]
        for t in (1, 4, 4)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_single_class_training_falls_back_to_a_constant_model():
    trainer = ct.ClassificationTrainer(n_runs=1)
    model = trainer._fit({}, pd.DataFrame({"a": [1.0, 2.0]}), np.array(["x", "x"]))
    assert list(model.predict(pd.DataFrame({"a": [0.0, 5.0, 9.0]}))) == ["x", "x", "x"]


# ── U3 ───────────────────────────────────────────────────────────────────


def test_delta_is_one_sided_and_paired_by_split():
    ttr = _result(TrainingStrategy.TTR, [0.80, 0.82, 0.78])
    better = _result(TrainingStrategy.TSTR, [0.85, 0.86, 0.84])
    worse = _result(TrainingStrategy.TSTR, [0.60, 0.64, 0.56])

    d_better, _ = mem._strategy_delta(better, ttr, "classification")
    d_worse, se = mem._strategy_delta(worse, ttr, "classification")

    assert d_better < 0  # beating TTR: no penalty once clipped
    assert d_worse == pytest.approx(0.20)
    assert se == pytest.approx(np.std([0.20, 0.18, 0.22], ddof=1) / math.sqrt(3))


def test_regression_delta_is_relative_and_one_sided():
    ttr = _result(TrainingStrategy.TTR, [2.0, 2.0])
    worse = _result(TrainingStrategy.TSTR, [3.0, 3.0])  # MAE 50 % higher
    d, _ = mem._strategy_delta(worse, ttr, "regression")
    assert d == pytest.approx(0.5)


def test_absolute_scale_runs_from_no_signal_to_ttr(monkeypatch):
    _fake_strategies(monkeypatch, {"ttr": [0.8, 0.8], "tstr": [0.9, 0.9]})
    metric = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), _context())
    # Symmetric (2026-10-19): 0.1 above TTR costs as much as 0.1 below it.
    assert metric.compute().value == pytest.approx(2 / 3)

    for tstr, expected in ((0.65, 0.5), (0.5, 0.0), (0.4, 0.0)):
        mem.clear_ttr_cache()
        _fake_strategies(monkeypatch, {"ttr": [0.8, 0.8], "tstr": [tstr, tstr]})
        res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), _context()).compute()
        assert res.value == pytest.approx(expected)
    assert "formula" in res.details
    assert res.details["strategies"]["tstr"]["delta_from_ttr"] == pytest.approx(0.4)


def test_regression_scale_uses_mae_of_the_median(monkeypatch):
    ttr = _result(TrainingStrategy.TTR, [1.0, 1.0])
    tstr = _result(TrainingStrategy.TSTR, [2.0, 2.0])
    scale, why = mem._utility_scale(
        ttr, _result(TrainingStrategy.TTR, []), [3.0, 3.0], "regression"
    )
    assert why is None
    u, _ = mem._absolute_utility(tstr, [3.0, 3.0], scale, "regression")
    assert u == pytest.approx(0.5)  # gains 1 of the 2 MAE units TTR gains


def test_half_sample_noise_moves_the_anchors_inwards(monkeypatch):
    """Another real sample scores 1 and a model without signal 0 (2026-10-10:
    an unseen real third of rrhh scored 0.54 with the anchors at the means)."""
    halves = [0.72, 0.70, 0.68]  # paired with TTR 0.8: -0.08, -0.10, -0.12, sigma 0.02
    sigma = 0.02
    upper = 0.3 - 1.6449 * math.sqrt(2) * sigma
    lower = 1.6449 * sigma
    for tstr, expected in (
        (0.5 + upper, 1.0),  # as far below TTR as a real sample can fall
        (0.5 + lower, 0.0),  # as far above chance as a useless model can rise
        (0.5 + (upper + lower) / 2, 0.5),
    ):
        mem.clear_ttr_cache()
        _fake_strategies(monkeypatch, {"ttr": [0.8] * 3, "half": halves, "tstr": [tstr] * 3})
        ctx = _context({"n_splits": 3})
        res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), ctx).compute()
        assert res.value == pytest.approx(expected, abs=1e-3)
    assert res.details["scale"]["sigma_source"] == "half-samples of the real rows"
    assert res.details["scale"]["sigma"] == pytest.approx(sigma)


def test_signal_within_three_noise_units_is_not_applicable(monkeypatch):
    # gain 0.3; paired sigma 0.1, noise 0.1414, 3 noise units 0.42 > 0.3
    # (until 2026-10-25 the threshold was 5 noise units, §21)
    _fake_strategies(monkeypatch, {"ttr": [0.8] * 3, "half": [0.80, 0.70, 0.60], "tstr": [0.8] * 3})
    ctx = _context({"n_splits": 3})
    res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), ctx).compute()
    assert res.status == "not_applicable"
    assert "3× its sampling noise" in res.details["reason"]


def test_half_samples_train_on_half_the_rows_and_test_on_the_holdout(monkeypatch):
    seen: list[tuple[int, int]] = []

    class _Trainer:
        def train_and_evaluate(self, X_train, X_test, y_train, y_test, seed):
            seen.append((len(X_train), len(X_test)))
            return {}, [0.5], 0.5, 0.0

    real, holdout = _table(40, 0), _table(20, 2)
    res = mem._half_sample_scores(real, real, "y", ["x1", "c"], _Trainer(), 0, 3, holdout)
    assert seen == [(20, 20)] * 3
    assert res.scores_per_run == [0.5, 0.5, 0.5]


def test_real_data_without_signal_gives_nan(monkeypatch):
    _fake_strategies(monkeypatch, {"ttr": [0.52, 0.48, 0.51], "tstr": [0.5, 0.5, 0.5]})
    ctx = _context({"n_splits": 3})
    res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), ctx).compute()
    assert math.isnan(res.value)
    assert res.status == "not_applicable"
    assert "no detectable signal" in res.details["reason"]


def test_no_signal_score_closed_form():
    # Classification is scored with the ROC AUC: a model without signal is 0.5
    # whatever the class balance (until 2026-10-19, weighted F1 of the best
    # trivial model: 0.9 * 1.8 / 1.9 for a 90/10 target).
    balanced = np.array([0, 1] * 50)
    assert mem._no_signal_score(balanced, balanced, "classification") == pytest.approx(0.5)
    skewed = np.array([0] * 90 + [1] * 10)
    assert mem._no_signal_score(skewed, skewed, "classification") == pytest.approx(0.5)
    assert mem._no_signal_score(
        np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0]), "regression"
    ) == pytest.approx(1.0)


def test_tstr_is_the_default_and_others_are_opt_in(monkeypatch):
    seen = _fake_strategies(
        monkeypatch, {"ttr": [0.8], "tstr": [0.7], "tts": [0.9], "trts": [0.9], "ttrs": [0.8]}
    )
    res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), _context()).compute()
    assert res.details["selected_strategies"] == ["tstr"]
    assert set(seen) == {TrainingStrategy.TTR, TrainingStrategy.TSTR}

    ctx = _context({"strategies": ["tstr", "tts"]})
    res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), ctx).compute()
    assert res.value == pytest.approx((2 / 3 + 2 / 3) / 2)  # TTS 0.9: 0.1 above TTR


def test_n_splits_is_configurable_and_reported(monkeypatch):
    assert "n_splits" in ML_CONFIG_KEYS
    calls: list[dict] = []
    original = mem._strategy_data

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(mem, "_strategy_data", spy)
    monkeypatch.setattr(mem, "_chance_scores", lambda *a, **k: [0.3] * k["n_splits"])
    monkeypatch.setattr(
        ct.CatBoostTrainer, "train_and_evaluate", lambda self, *a, **k: ({}, [0.5], 0.5, 0.0)
    )
    ctx = _context({"n_splits": 3})
    res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), ctx).compute()

    assert res.details["n_splits"] == 3
    # TTR, TTR on a half-sample and TSTR
    assert sorted(c["seed"] for c in calls) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert len(res.details["ttr_scores_per_split"]) == 3
    assert "score_se" in res.details


def test_n_splits_must_be_positive():
    ctx = _context({"n_splits": 0})
    with pytest.raises(ValueError, match="n_splits"):
        mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), ctx)


def test_holdout_is_the_test_set_of_the_real_tested_strategies():
    real, synth, holdout = _table(40, 0), _table(30, 1), _table(20, 2)
    feats = ["x1", "c"]
    for strategy, n_train in (
        (TrainingStrategy.TTR, 40),
        (TrainingStrategy.TSTR, 30),
        (TrainingStrategy.TTRS, 70),
    ):
        data = mem._strategy_data(strategy, real, synth, "y", feats, seed=0, holdout=holdout)
        assert len(data["X_train"]) == n_train
        pd.testing.assert_frame_equal(data["X_test"], holdout[feats])


# ── U5 ───────────────────────────────────────────────────────────────────


def test_float_target_is_mapped_not_truncated():
    real = pd.DataFrame({"x": range(6), "y": [0.3, 0.7, 1.0, 0.3, 0.7, 1.0]})
    synth = pd.DataFrame({"x": range(4), "y": [0.7, 0.3, 1.0, 2.5]})

    r, s, _, n_classes, unseen = mem._encode_classification_labels(real, synth, "y")

    assert n_classes == 3  # astype(int) merged 0.3 and 0.7 into 0
    mapping = dict(zip(real["y"], r["y"]))
    assert list(s["y"][:3]) == [mapping[0.7], mapping[0.3], mapping[1.0]]
    assert s["y"].iloc[3] == 3  # unseen label: its own extra class
    assert unseen == {"synthetic.y": 1}


def test_equal_labels_of_different_types_share_a_class():
    real = pd.DataFrame({"y": [1, 0, 1]})
    synth = pd.DataFrame({"y": ["1", 0.0, "1.0"]})
    r, s, _, _, unseen = mem._encode_classification_labels(real, synth, "y")
    assert list(s["y"]) == [r["y"].iloc[0], r["y"].iloc[1], r["y"].iloc[0]]
    assert unseen == {}


# ── U6 ───────────────────────────────────────────────────────────────────


def test_failed_principal_strategy_gives_nan(monkeypatch):
    _fake_strategies(monkeypatch, {"ttr": [0.8], "tstr": None})
    res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), _context()).compute()
    assert math.isnan(res.value)
    assert "tstr" in res.details["error"]
    assert res.details["failed_strategies"] == {"tstr": "boom"}


def test_failed_secondary_strategy_is_listed(monkeypatch):
    _fake_strategies(monkeypatch, {"ttr": [0.8], "tstr": [0.8], "trts": None})
    ctx = _context({"strategies": ["tstr", "trts"]})
    res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), ctx).compute()
    assert res.value == pytest.approx(1.0)
    assert res.details["failed_strategies"] == {"trts": "boom"}


def test_failed_ttr_gives_nan(monkeypatch):
    _fake_strategies(monkeypatch, {"ttr": None, "tstr": [0.8]})
    res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), _context()).compute()
    assert math.isnan(res.value)
    assert "TTR" in res.details["error"]


# ── U7 ───────────────────────────────────────────────────────────────────


def test_classification_efficiency_uses_the_shared_code(monkeypatch):
    _fake_strategies(monkeypatch, {"ttr": [0.8], "tstr": [0.6]})
    m = ClassificationEfficiencyMetric().fit(_table(40, 0), _table(40, 1), _context())
    res = m.compute()
    assert res.id == "utility.classification_efficiency"
    assert res.value == pytest.approx(1 / 3)
    assert res.details["task_type"] == "classification"


def test_regression_efficiency_rejects_a_categorical_target(monkeypatch):
    _fake_strategies(monkeypatch, {"ttr": [0.8], "tstr": [0.6]})
    m = RegressionEfficiencyMetric().fit(_table(40, 0), _table(40, 1), _context())
    res = m.compute()
    assert res.id == "utility.regression_efficiency"
    assert math.isnan(res.value)
    assert "not numeric" in res.details["error"]


def test_context_target_is_used_without_dataset_spec(monkeypatch):
    _fake_strategies(monkeypatch, {"ttr": [0.8], "tstr": [0.8]})
    ctx = _context(target="y")
    ctx["dataset_spec"] = None
    res = mem.MLEfficiencyMetric().fit(_table(40, 0), _table(40, 1), ctx).compute()
    assert res.value == pytest.approx(1.0)


def test_multi_target_classification_accepts_categorical_features():
    """``MultiOutputClassifier`` could not clone CatBoost with ``cat_features``."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "c": rng.choice(["u", "v"], size=60)})
    y = np.column_stack([(X["a"] > 0).astype(int), (X["c"] == "u").astype(int)])
    trainer = ct.MultiTargetClassificationTrainer(n_runs=1, thread_count=1)
    _, scores, _, _ = trainer.train_and_evaluate(X[:45], X[45:], y[:45], y[45:], seed=0)
    assert 0.0 <= scores[0] <= 1.0


def test_regression_is_trained_with_the_mae_it_is_scored_with():
    assert ct.RegressionTrainer(n_runs=1)._base_params(0)["loss_function"] == "MAE"
    # Multi-target regression keeps MultiRMSE (CatBoost has no multi-output MAE).
    assert ct.MultiTargetRegressionTrainer(n_runs=1)._base_params(0)["loss_function"] == "MultiRMSE"
