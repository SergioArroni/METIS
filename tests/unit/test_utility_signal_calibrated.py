"""Utility applicability decided once per real table by the calibration (2026-10-18).

With five half-samples σ is uncertain by ~35 % and the "not applicable" of
``utility.ml_efficiency`` flipped with the seed on rrhh (4 of 5 seeds of the
2026-10-10 benchmark).  The calibration now measures the signal on 4 splits
× 5 half-samples, stores the decision with the bounds, and every evaluation
with those bounds follows it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metis.application.orchestrator import Orchestrator
from metis.calibrate import CalibrationBounds, MetricCalibrator
from metis.infrastructure.metrics.utility import ml_efficiency_metric as mem
from metis.interface.sdk import Evaluator

SCHEMA = {"x": "continuous", "c": "nominal", "y": "nominal"}


def _table(n: int, seed: int, signal: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x + rng.normal(0, 0.5, n) > 0 if signal else rng.random(n) > 0.5
    return pd.DataFrame({"x": x, "c": rng.choice(list("abc"), n), "y": np.where(y, "yes", "no")})


def _config(tmp_path, **extra) -> tuple[str, dict]:
    cfg = {
        "data": {"schema": dict(SCHEMA), "target": "y"},
        "metrics": ["fidelity.ks", "utility.ml_efficiency"],
        "evaluation": {"n_runs": 1},
        "reproducibility": {"seed": 0},
        **extra,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path), cfg


def _calibrate(tmp_path, real: pd.DataFrame) -> CalibrationBounds:
    cfg_path, _ = _config(tmp_path)
    return MetricCalibrator().calibrate(real, cfg_path, n_iterations=2, sample_size=150)


# ── the decision is measured by the calibration ──────────────────────────


def test_calibration_finds_signal_with_twenty_half_samples(tmp_path):
    bounds = _calibrate(tmp_path, _table(450, 0))
    signal = bounds.utility_signal
    assert signal["applicable"] is True
    assert signal["n_half_samples"] == 20 and signal["sigma_dof"] == 16
    assert signal["ttr_gain"] > signal["threshold"] > 0
    assert signal["rows_per_part"] == 90  # 450 rows, 60/20/20 (min(150, 450 // 5))

    path = tmp_path / "b.json"
    bounds.save(str(path))
    assert CalibrationBounds.load(str(path)).utility_signal == signal


def test_calibration_finds_no_signal_on_a_random_target(tmp_path):
    signal = _calibrate(tmp_path, _table(450, 0, signal=False)).utility_signal
    assert signal["applicable"] is False
    assert "no detectable signal" in signal["reason"]


def test_calibration_without_utility_metric_measures_nothing(tmp_path):
    cfg_path, cfg = _config(tmp_path)
    cfg["metrics"] = ["fidelity.ks"]
    with Path(cfg_path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    bounds = MetricCalibrator().calibrate(_table(300, 0), cfg_path, n_iterations=2)
    assert bounds.utility_signal is None


# ── the evaluation follows it ────────────────────────────────────────────


def _evaluate(tmp_path, signal: dict | None, real: pd.DataFrame) -> dict:
    bounds = CalibrationBounds()
    bounds.set_bounds("fidelity.ks", lower_bound=0.0, upper_bound=1.0)
    bounds.utility_signal = signal
    path = tmp_path / "bounds.json"
    bounds.save(str(path))
    _, cfg = _config(tmp_path, calibration={"bounds_file": str(path)})
    summary = Orchestrator().evaluate_dataframes(
        real, _table(150, 7), cfg, seed=0, holdout=_table(150, 8)
    )
    (result,) = [r for r in summary.results if r.id == "utility.ml_efficiency"]
    return result


_SIGNAL = {"applicable": True, "reason": None, "ttr_gain": 0.3, "threshold": 0.05}


def test_not_applicable_decided_by_the_calibration(tmp_path):
    no = {**_SIGNAL, "applicable": False, "reason": "The real data carries no signal"}
    result = _evaluate(tmp_path, no, _table(150, 0))
    assert result.status == "not_applicable"
    assert "decided once for this real table" in result.details["reason"]


def test_calibrated_signal_overrides_the_run_decision(tmp_path, monkeypatch):
    # The run alone would call it not applicable (threshold raised); the
    # calibration's decision wins and the score is computed.
    monkeypatch.setattr(mem, "UTILITY_SIGNAL_MULTIPLE", 1e6)
    result = _evaluate(tmp_path, _SIGNAL, _table(150, 0))
    assert result.status == "ok" and math.isfinite(result.value)
    assert result.details["signal_decision"]["source"] == "calibration"

    alone = _evaluate(tmp_path, None, _table(150, 0))
    assert alone.status == "not_applicable"


# ── pooling ──────────────────────────────────────────────────────────────


def _probe(ttr: list[float], half: list[float]) -> dict:
    return {
        "task_type": "classification",
        "no_signal_scores_per_split": [0.5] * len(ttr),
        "ttr_scores_per_split": ttr,
        "ttr_half_sample_scores_per_split": half,
    }


def test_pooled_sigma_is_within_split():
    # Each split loses a different amount when halved (0.02 vs 0.10): that
    # offset is not sampling noise of one run and must not inflate σ.
    a = _probe([0.80] * 5, [0.78 + d for d in (-0.01, 0, 0.01, 0, 0)])
    b = _probe([0.80] * 5, [0.70 + d for d in (-0.01, 0, 0.01, 0, 0)])
    signal = mem.pooled_utility_signal([a, b])
    assert signal["sigma"] == pytest.approx(np.std([-0.01, 0, 0.01, 0, 0], ddof=1))
    assert signal["n_half_samples"] == 10 and signal["sigma_dof"] == 8
    assert signal["ttr_gain"] == pytest.approx(0.30)
    assert signal["applicable"] is True


def test_pooled_signal_needs_two_pairs_in_some_split():
    assert mem.pooled_utility_signal([_probe([0.8], [0.7])]) is None


# ── without calibration: measured once through the SDK ───────────────────


def test_sdk_measures_the_signal_once_and_evaluations_follow_it(tmp_path, monkeypatch):
    _, cfg = _config(tmp_path)
    real, holdout = _table(200, 0), _table(150, 8)
    ev = Evaluator()
    signal = ev.measure_utility_signal(real, cfg, holdout=holdout)
    assert signal["applicable"] is True and signal["n_half_samples"] == 20
    assert signal["source"] == "measure_utility_signal"

    monkeypatch.setattr(mem, "UTILITY_SIGNAL_MULTIPLE", 1e6)  # the run alone: not applicable
    summary = ev.evaluate(real, _table(150, 7), cfg, holdout=holdout, utility_signal=signal)
    (result,) = [r for r in summary.results if r.id == "utility.ml_efficiency"]
    assert result.status == "ok" and math.isfinite(result.value)

    no_target = {**cfg, "data": {"schema": dict(SCHEMA)}}
    assert ev.measure_utility_signal(real, no_target, holdout=holdout) is None


def test_table_level_anchors_replace_the_runs_ttr_on_tables_of_the_same_size(tmp_path):
    _, cfg = _config(tmp_path)
    real, holdout = _table(200, 0), _table(150, 8)
    ev = Evaluator()
    signal = ev.measure_utility_signal(real, cfg, holdout=holdout)
    assert signal["rows"] == {"real": 200, "holdout": 150}
    assert signal["lower"] < signal["upper"]

    def utility(real_table):
        # Synthetic of the same size: real and synthetic are compared at equal size.
        summary = ev.evaluate(
            real_table, _table(len(real_table), 7), cfg, holdout=holdout, utility_signal=signal
        )
        (result,) = [r for r in summary.results if r.id == "utility.ml_efficiency"]
        return result

    same = utility(real)
    assert same.status == "ok"
    assert "ttr_baseline" not in same.details  # no TTR trained by the run
    assert same.details["scale"]["lower"] == signal["lower"]
    assert same.details["scale"]["upper"] == signal["upper"]
    assert same.details["scale"]["sigma_source"].startswith("table level")

    # Another size: the decision holds, the anchors are the run's own.
    other = utility(_table(300, 0))
    assert "ttr_baseline" in other.details
    assert other.details["scale"]["upper"] != signal["upper"]


def test_evaluating_tables_larger_than_the_calibration_thirds_warns():
    bounds = CalibrationBounds()
    bounds.set_metadata("dataset_rows", 3000)
    bounds.set_metadata("sample_size", 3000)
    warn = Orchestrator._calibration_size_warning
    assert warn(bounds, 600, 600) == []  # the benchmark: the 20 % part on both sides
    (message,) = warn(bounds, 3000, 3000)  # the whole table against its 20 % parts
    assert "optimistic" in message and "600 rows" in message
