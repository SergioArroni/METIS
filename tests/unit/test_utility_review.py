"""Second review of the utility metrics (2026-10-09).

* R-U1: TTRS changed the real test rows when the synthetic table was smaller.
* R-U2: forced classification rejected a binary target on tables under 20 rows.
* R-U3: invalid ``thread_count`` / ``METIS_N_JOBS`` surfaced as a NaN utility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.utility import ml_efficiency_metric as mem
from metis.infrastructure.metrics.utility.ml_common import MLMetricMixin
from metis.infrastructure.metrics.utility.ml_efficiency import catboost_trainer as ct
from metis.infrastructure.metrics.utility.ml_efficiency.ttr import TTRMetric
from metis.infrastructure.metrics.utility.ml_efficiency.ttrs import TTRSMetric


def _table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "c": rng.choice(["a", "b", "c"], n),
            "y": np.where(x + rng.normal(0, 1, n) > 0, "si", "no"),
        }
    )


# ── R-U1: TTRS keeps TTR's real test rows ────────────────────────────────


def test_ttrs_tests_on_the_ttr_rows_when_the_synthetic_table_is_smaller():
    real, synth = _table(300, 0), _table(100, 1)
    features = ["x", "c"]
    for seed in (0, 1):
        ttr = TTRMetric().prepare_data(real, synth, "y", features, seed=seed)
        ttrs = TTRSMetric().prepare_data(real, synth, "y", features, seed=seed)
        pd.testing.assert_frame_equal(ttr["X_test"], ttrs["X_test"])
        np.testing.assert_array_equal(ttr["y_test"], ttrs["y_test"])
        assert len(ttrs["X_train"]) == len(ttr["X_train"]) + 80


# ── R-U2 / R-U3: small tables and configuration ──────────────────────────


class _Metric:
    def __init__(self, real):
        self._real_data = real


def test_forced_classification_accepts_a_binary_target_on_a_tiny_table():
    real = pd.DataFrame({"y": ["a", "b"] * 7})
    assert mem._check_forced_task(_Metric(real), "y", "classification") is None


def test_forced_classification_still_rejects_a_continuous_target():
    real = pd.DataFrame({"y": np.linspace(0, 1, 300)})
    assert "too many" in mem._check_forced_task(_Metric(real), "y", "classification")


@pytest.mark.parametrize("value", [0, -2])
def test_invalid_thread_count_is_a_config_error(value):
    class _M(MLMetricMixin):
        pass

    ctx = {"config": {"evaluation": {"ml_efficiency_config": {"thread_count": value}}}}
    with pytest.raises(ValueError, match="thread_count"):
        _M()._apply_ml_config(ctx)


def test_malformed_metis_n_jobs_does_not_break_training(monkeypatch):
    monkeypatch.setenv("METIS_N_JOBS", "cuatro")
    assert ct.CatBoostTrainer._thread_budget() == 1


# ── R-U5: ordinals count once and their shadow never leaks the target ───


class _Ctx(MLMetricMixin):
    def __init__(self, real, target):
        self._real_data = real
        self._synth_data = real
        self._context = {"target": target}


def test_feature_columns_keep_both_views_of_ordinals_but_drop_the_target_rank():
    real = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "nivel": ["bajo", "alto"],
            "nivel__ord": [0.0, 1.0],
            "y": ["a", "b"],
            "y__ord": [0.0, 1.0],
        }
    )
    assert _Ctx(real, "y")._get_feature_cols("y") == ["x", "nivel", "nivel__ord"]


def test_datetime_target_is_found_under_its_cast_name():
    real = pd.DataFrame({"x": [1.0, 2.0], "fecha__ts": [1.0, 2.0]})
    targets, error = _Ctx(real, "fecha")._validate_target()
    assert error is None
    assert targets == ["fecha__ts"]
