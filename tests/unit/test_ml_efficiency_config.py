"""Run-time configuration of the utility metrics.

Guards three regressions found while using METIS as a scorer in a loop:

* ``evaluation.ml_efficiency_config`` never reached the metrics (the
  evaluation context carried no ``config``), so every call paid the
  default budget.
* CatBoost was built without ``thread_count`` and opened every core even
  with ``METIS_N_JOBS=1``, and wrote ``catboost_info/`` into the cwd.
* The SDK ignored ``reproducibility.seed`` and always used 42.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.utility import ml_efficiency_metric as mem
from metis.infrastructure.metrics.utility.ml_common import MLMetricMixin
from metis.infrastructure.metrics.utility.ml_efficiency import catboost_trainer as ct
from metis.infrastructure.metrics.utility.ml_efficiency.base import TrainingStrategy
from metis.interface.sdk import Evaluator


def _table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    c = rng.choice(["a", "b", "c"], size=n)
    y = np.where(x1 + 0.5 * x2 + (c == "a") > 0, "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "c": c, "y": y})


# Utility needs a holdout (real rows the generator never saw) since 2026-10-19.
_HOLDOUT = _table(200, 77)


def _config(ml_config: dict | None = None, seed: int = 0) -> dict:
    cfg: dict = {
        "data": {
            "schema": {
                "x1": "continuous",
                "x2": "continuous",
                "c": "nominal",
                "y": "nominal",
            },
            "target": "y",
        },
        "metrics": ["utility.ml_efficiency"],
        "evaluation": {"n_runs": 1},
        "reproducibility": {"seed": seed},
    }
    if ml_config is not None:
        cfg["evaluation"]["ml_efficiency_config"] = ml_config
    return cfg


def _ml_result(summary):
    (result,) = [r for r in summary.results if r.id == "utility.ml_efficiency"]
    return result


@pytest.fixture(autouse=True)
def _clean_ttr_cache():
    mem.clear_ttr_cache()
    yield
    mem.clear_ttr_cache()


# ── B1: the config reaches the metric ────────────────────────────────────


def test_ml_efficiency_config_reaches_the_metric():
    # Big enough for the real signal to clear 3× its sampling noise.
    real, synth = _table(400, 0), _table(400, 1)
    cfg = _config({"n_splits": 2, "n_runs": 1, "strategies": ["tstr"]})

    result = _ml_result(Evaluator().evaluate(real, synth, cfg, holdout=_HOLDOUT))

    assert "error" not in result.details, result.details
    assert result.details["n_splits"] == 2
    assert result.details["n_runs"] == 1
    assert result.details["selected_strategies"] == ["tstr"]
    assert set(result.details["strategies"]) == {"tstr"}


def test_unknown_ml_efficiency_key_is_reported_not_ignored():
    real, synth = _table(60, 0), _table(60, 1)
    cfg = _config({"n_trails": 2})  # typo

    result = _ml_result(Evaluator().evaluate(real, synth, cfg, holdout=_HOLDOUT))

    assert np.isnan(result.value)
    assert "n_trails" in result.details["error"]


def test_n_trials_is_accepted_with_a_warning_and_ignored():
    """There is no hyperparameter search any more: old configs keep working."""
    ctx = {"config": {"evaluation": {"ml_efficiency_config": {"n_trials": 10}}}}
    with pytest.warns(DeprecationWarning, match="n_trials"):
        MLMetricMixin()._apply_ml_config(ctx)


# ── B2: CatBoost thread budget and side effects ──────────────────────────


class _Recorder:
    """Stands in for a CatBoost estimator and records its constructor params."""

    calls: list[dict] = []

    def __init__(self, **params):
        _Recorder.calls.append(params)

    def fit(self, X, y, **fit_kwargs):
        # Trials pass ``eval_set``/``early_stopping_rounds`` like real CatBoost.
        self._label = y[0]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        label = np.asarray(self._label)
        return np.repeat(label[None, ...], len(X), axis=0)

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))


def _run_trainer(monkeypatch, thread_count=None):
    _Recorder.calls = []
    monkeypatch.setattr(ct.ClassificationTrainer, "_get_model_class", lambda self: _Recorder)
    trainer = ct.ClassificationTrainer(n_runs=2, thread_count=thread_count)
    X = pd.DataFrame({"a": np.arange(20.0), "b": ["x", "y"] * 10})
    y = np.array([0, 1] * 10)
    trainer.train_and_evaluate(X[:15], X[15:], y[:15], y[15:], seed=0)
    return _Recorder.calls


def test_catboost_threads_follow_metis_n_jobs(monkeypatch):
    monkeypatch.setenv("METIS_N_JOBS", "1")
    calls = _run_trainer(monkeypatch)

    assert len(calls) == 3  # 1 early-stopping fit + 2 final runs
    assert {c["thread_count"] for c in calls} == {1}
    assert all(c["allow_writing_files"] is False for c in calls)


def test_every_fit_gets_the_whole_thread_budget(monkeypatch):
    """Fits run one after another; CatBoost gives the same model with 1 or N threads."""
    monkeypatch.setenv("METIS_N_JOBS", "3")
    calls = _run_trainer(monkeypatch)

    assert {c["thread_count"] for c in calls} == {3}


def test_without_metis_n_jobs_catboost_keeps_its_default(monkeypatch):
    monkeypatch.delenv("METIS_N_JOBS", raising=False)
    calls = _run_trainer(monkeypatch)

    assert {c["thread_count"] for c in calls} == {-1}


def test_explicit_thread_count_wins(monkeypatch):
    monkeypatch.setenv("METIS_N_JOBS", "8")
    calls = _run_trainer(monkeypatch, thread_count=2)

    assert {c["thread_count"] for c in calls} == {2}


def test_final_runs_keep_the_fixed_params(monkeypatch):
    """Without early stopping (no best iteration) the final fits keep the defaults."""
    monkeypatch.setenv("METIS_N_JOBS", "1")
    calls = _run_trainer(monkeypatch)

    final = calls[-1]
    assert final["iterations"] == ct.MAX_ITERATIONS
    assert final["one_hot_max_size"] == ct.ONE_HOT_MAX_SIZE
    assert final["verbose"] is False


def test_multi_target_regression_keeps_multirmse_in_final_runs(monkeypatch):
    _Recorder.calls = []
    monkeypatch.setattr(ct.MultiTargetRegressionTrainer, "_get_model_class", lambda self: _Recorder)
    monkeypatch.setattr(ct.MultiTargetRegressionTrainer, "_compute_metric", lambda self, a, b: 0.0)
    trainer = ct.MultiTargetRegressionTrainer(n_runs=1)
    X = pd.DataFrame({"a": np.arange(20.0)})
    y = np.column_stack([np.arange(20.0), np.arange(20.0)])
    trainer.train_and_evaluate(X[:15], X[15:], y[:15], y[15:], seed=0)

    assert _Recorder.calls[-1]["loss_function"] == "MultiRMSE"


# ── B3: TTR is computed once per reference table ─────────────────────────


def _count_ttr(monkeypatch) -> list[TrainingStrategy]:
    seen: list[TrainingStrategy] = []
    original = mem._evaluate_strategy

    def spy(strategy, *args, **kwargs):
        seen.append(strategy)
        return original(strategy, *args, **kwargs)

    monkeypatch.setattr(mem, "_evaluate_strategy", spy)
    return seen


def test_ttr_is_reused_across_calls_with_the_same_reference(monkeypatch):
    seen = _count_ttr(monkeypatch)
    real = _table(400, 0)
    cfg = _config({"n_runs": 1, "strategies": ["tstr"]})

    first = _ml_result(Evaluator().evaluate(real, _table(400, 1), cfg, holdout=_HOLDOUT))
    second = _ml_result(Evaluator().evaluate(real, _table(400, 2), cfg, holdout=_HOLDOUT))

    assert seen.count(TrainingStrategy.TTR) == 1
    assert seen.count(TrainingStrategy.TSTR) == 2
    assert first.details["ttr_baseline"] == second.details["ttr_baseline"]


def test_ttr_cache_misses_on_other_reference_or_budget(monkeypatch):
    seen = _count_ttr(monkeypatch)
    cfg1 = _config({"n_runs": 1, "strategies": ["tstr"]})
    cfg2 = _config({"n_runs": 2, "strategies": ["tstr"]})

    Evaluator().evaluate(_table(60, 0), _table(60, 1), cfg1, holdout=_HOLDOUT)
    Evaluator().evaluate(_table(60, 3), _table(60, 1), cfg1, holdout=_HOLDOUT)  # other reference
    Evaluator().evaluate(_table(60, 0), _table(60, 1), cfg2, holdout=_HOLDOUT)  # other budget

    assert seen.count(TrainingStrategy.TTR) == 3


def test_ttr_cache_can_be_disabled(monkeypatch):
    seen = _count_ttr(monkeypatch)
    real = _table(60, 0)
    cfg = _config({"n_runs": 1, "strategies": ["tstr"], "cache_ttr": False})

    Evaluator().evaluate(real, _table(60, 1), cfg, holdout=_HOLDOUT)
    Evaluator().evaluate(real, _table(60, 2), cfg, holdout=_HOLDOUT)

    assert seen.count(TrainingStrategy.TTR) == 2


# ── SDK seed ─────────────────────────────────────────────────────────────


def test_sdk_uses_the_config_seed():
    cfg = _config({"n_runs": 1, "strategies": ["tstr"]}, seed=7)
    summary = Evaluator().evaluate(_table(60, 0), _table(60, 1), cfg, holdout=_HOLDOUT)
    assert summary.artifacts["seed"] == 7


def test_sdk_explicit_seed_overrides_the_config():
    cfg = _config({"n_runs": 1, "strategies": ["tstr"]}, seed=7)
    summary = Evaluator().evaluate(_table(60, 0), _table(60, 1), cfg, seed=11, holdout=_HOLDOUT)
    assert summary.artifacts["seed"] == 11
