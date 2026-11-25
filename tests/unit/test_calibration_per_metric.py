"""Per-metric calibration (decision of 2026-10-09, study ``calib_v2``).

* privacy: upper = unseen real third, lower = exact copy of the training
  third, both measured with the holdout third;
* fidelity: upper = unseen real third, lower = uniform noise;
* utility: never calibrated;
* a metric whose anchors are within 5× their noise is excluded;
* old family-level bounds files are rejected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metis.application.aggregator import Aggregator
from metis.application.orchestrator import Orchestrator
from metis.calibrate import CalibrationBounds, MetricCalibrator
from metis.calibrate.core.bounds import NOISE_MULTIPLE
from metis.calibrate.strategies.base import tercile_split
from metis.calibrate.utils.evaluator import InMemoryEvaluator
from metis.domain.entities import MetricResult
from metis.interface import cli
from metis.shared.anchors import WARNING_Z

SCHEMA = {"x": "continuous", "z": "continuous", "c": "nominal", "y": "nominal"}
METRICS = ["fidelity.ks", "fidelity.tvd", "privacy.dcr", "privacy.mia", "utility.ml_efficiency"]


def _table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "z": x + rng.normal(scale=0.5, size=n),
            "c": rng.choice(list("abc"), size=n),
            "y": np.where(x > 0, "yes", "no"),
        }
    )


def _config_file(tmp_path, **extra) -> str:
    cfg = {
        "data": {"schema": dict(SCHEMA), "target": "y"},
        "metrics": list(METRICS),
        "evaluation": {"n_runs": 1},
        "reproducibility": {"seed": 0},
        **extra,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cal")
    cfg_path = _config_file(tmp)
    real = _table(600, 0)
    bounds = MetricCalibrator().calibrate(real, cfg_path, n_iterations=3, sample_size=200)
    return real, cfg_path, bounds


def test_thirds_are_disjoint_and_repeatable():
    real = _table(90, 0).reset_index().rename(columns={"index": "row"})
    a = tercile_split(real, 1000, seed=3)
    b = tercile_split(real, 1000, seed=3)
    rows = [set(part["row"]) for part in a]
    assert [len(r) for r in rows] == [54, 18, 18]  # 60/20/20 (thirds until 2026-10-19)
    assert not (rows[0] & rows[1] or rows[0] & rows[2] or rows[1] & rows[2])
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)


def test_calibration_split_is_the_benchmark_split():
    """Same function as the benchmark: stratified by the target, 3m / m / m rows."""
    from metis.shared.splits import split_real

    rng = np.random.default_rng(1)
    real = pd.DataFrame({"row": range(500), "y": np.where(rng.random(500) < 0.2, "yes", "no")})
    parts = tercile_split(real, 1000, seed=7, stratify_by="y")
    assert [len(p) for p in parts] == [300, 100, 100]
    for x, y in zip(parts, split_real(real, 7, stratify_by="y"), strict=True):
        pd.testing.assert_frame_equal(x, y)
    share = (real["y"] == "yes").mean()
    assert all(abs((p["y"] == "yes").mean() - share) < 0.011 for p in parts)
    small = tercile_split(real, 40, seed=7, stratify_by="y")
    assert [len(p) for p in small] == [120, 40, 40]
    assert all(abs((p["y"] == "yes").mean() - share) < 0.03 for p in small)
    thirds = tercile_split(real, 1000, seed=7, stratify_by="y", fractions=(1 / 3, 1 / 3, 1 / 3))
    assert [len(p) for p in thirds] == [166, 166, 166]


def test_calibration_seeds_never_reuse_an_evaluation_seed():
    from metis.calibrate.strategies.base import calibration_seed

    assert {calibration_seed(42, i) for i in range(10)}.isdisjoint(range(42, 52))


def test_calibration_evaluates_no_utility_metric(tmp_path):
    evaluator = InMemoryEvaluator(_config_file(tmp_path))
    assert evaluator.metrics_of(("fidelity", "privacy")) == [
        "fidelity.ks",
        "fidelity.tvd",
        "privacy.dcr",
        "privacy.mia",
    ]


def test_bounds_are_per_metric_with_the_approved_references(calibrated):
    _, _, bounds = calibrated
    assert bounds.get_all_families() == ["fidelity", "privacy"]
    assert not [m for m in bounds.get_all_metrics() if m.startswith("utility")]
    # Privacy: unseen real ≈ private (1); the copy of as many training rows as the
    # synthetic table has (a fifth of the table, a third of the training rows) leaks.
    # With 3 iterations the anchor, mean − 1.645 sd, moves with the seeds: the
    # upper DCR anchor of this table ranges 0.71–1.0 over base seeds 0, 7, 42, 1000.
    for metric in ("privacy.dcr", "privacy.mia"):
        lower, upper = bounds.get_bounds(metric)
        assert upper > 0.65, (metric, upper)
        assert lower < 0.25, (metric, lower)
        assert "copy" in bounds.bounds[metric]["reference_lower"]
    # Fidelity: noise far below unseen real.
    # On this 600-row toy with 3 iterations, whether they clear the 5× noise rule
    # depends on the seeds, so only the order of the anchors is checked here.
    for metric in ("fidelity.ks", "fidelity.tvd"):
        lower, upper = bounds.get_bounds(metric)
        assert upper > lower, (metric, lower, upper)
        assert "noise" in bounds.bounds[metric]["reference_lower"]
    entry = bounds.bounds["privacy.dcr"]
    assert entry["n_lower"] == entry["n_upper"] == 3
    assert {"lower_sd", "upper_sd", "noise", "calibratable"} <= set(entry)
    assert bounds.get_metadata("method") == "per_metric"


def test_bounds_round_trip_and_old_layout_is_rejected(calibrated, tmp_path):
    _, _, bounds = calibrated
    path = tmp_path / "b.json"
    bounds.save(str(path))
    loaded = CalibrationBounds.load(str(path))
    assert loaded.get_bounds("privacy.dcr") == pytest.approx(bounds.get_bounds("privacy.dcr"))

    old = tmp_path / "old.json"
    old.write_text(
        json.dumps({"bounds": {"fidelity": {"lower": 0.1, "upper": 0.9}}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="family-level"):
        CalibrationBounds.load(str(old))


def test_anchors_sit_at_the_95_quantile_of_their_iterations():
    """An unseen real sample scores 1 and noise 0, not only on average."""
    rng = np.random.default_rng(0)
    upper = list(0.9 + 0.01 * rng.standard_normal(10))
    lower = list(0.2 + 0.01 * rng.standard_normal(10))
    bounds = CalibrationBounds()
    bounds.set_bounds("fidelity.ks", lower_iterations=lower, upper_iterations=upper)
    l, u = bounds.get_bounds("fidelity.ks")
    assert l == pytest.approx(np.mean(lower) + 1.6449 * np.std(lower, ddof=1), abs=1e-4)
    assert u == pytest.approx(np.mean(upper) - 1.6449 * np.std(upper, ddof=1), abs=1e-4)
    assert bounds.bounds["fidelity.ks"]["upper_mean"] == pytest.approx(np.mean(upper))
    fresh_real = 0.9 + 0.01 * rng.standard_normal(200)
    fresh_noise = 0.2 + 0.01 * rng.standard_normal(200)
    real_scores = [bounds.normalize_with_bounds("fidelity.ks", v) for v in fresh_real]
    noise_scores = [bounds.normalize_with_bounds("fidelity.ks", v) for v in fresh_noise]
    assert np.mean(real_scores) > 0.995
    assert np.mean(noise_scores) < 0.005


def test_format_2_bounds_are_rebuilt_from_their_iterations(tmp_path):
    entry = {
        "family": "fidelity",
        "lower": 0.2,
        "upper": 0.9,
        "calibratable": True,
        "lower_iterations": [0.19, 0.21],
        "upper_iterations": [0.89, 0.91],
    }
    path = tmp_path / "v2.json"
    path.write_text(
        json.dumps({"format": 2, "bounds": {"fidelity.ks": entry}, "metadata": {"x": 1}}),
        encoding="utf-8",
    )
    loaded = CalibrationBounds.load(str(path))
    l, u = loaded.get_bounds("fidelity.ks")
    assert 0.2 < l < u < 0.9
    assert loaded.get_metadata("rebuilt_from_format") == 2
    assert loaded.get_metadata("x") == 1


def test_metric_within_its_noise_is_excluded_from_the_aggregation():
    bounds = CalibrationBounds()
    bounds.set_bounds("fidelity.ks", lower_iterations=[0.1, 0.12], upper_iterations=[0.9, 0.92])
    # range 0.01 with noise ≈0.02: cannot tell its anchors apart
    bounds.set_bounds("fidelity.tvd", lower_iterations=[0.95, 0.97], upper_iterations=[0.96, 0.98])
    assert bounds.is_calibratable("fidelity.ks")
    assert not bounds.is_calibratable("fidelity.tvd")
    assert f"{NOISE_MULTIPLE:g}×noise" in bounds.exclusion_reason("fidelity.tvd")

    results = [
        MetricResult(id="fidelity.ks", value=0.51, details={}, family="fidelity"),
        MetricResult(id="fidelity.tvd", value=0.2, details={}, family="fidelity"),
        MetricResult(id="utility.ml_efficiency", value=0.4, details={}, family="utility"),
    ]
    out = Aggregator(calibration_bounds=bounds).aggregate(results, None)
    assert out["calibration_method"] == "per_metric"
    assert out["calibrated_metrics"]["fidelity.ks"] == pytest.approx(0.5)
    assert "fidelity.tvd" in out["calibration_excluded"]
    assert out["fidelity_score"] == pytest.approx(0.5)  # tvd left out, not averaged in
    assert out["utility_score"] == pytest.approx(0.4)  # utility untouched
    assert out["fidelity_score_raw"] == pytest.approx(np.median([0.51, 0.2]))  # traceability
    assert results[0].value == 0.51  # caller's results are not modified


def test_metric_far_below_another_real_sample_is_warned_without_changing_the_score():
    """Decision §24: warning at WARNING_Z sd of the upper reference, outside the score."""
    bounds = CalibrationBounds()
    upper = [0.88, 0.90, 0.92]  # mean 0.90, sd 0.02
    bounds.set_bounds("fidelity.ks", lower_iterations=[0.1, 0.12], upper_iterations=upper)
    bounds.set_bounds("fidelity.tvd", lower_iterations=[0.1, 0.12], upper_iterations=upper)
    assert bounds.deviation_from_real("fidelity.ks", 0.78) == pytest.approx(6.0)
    assert bounds.deviation_from_real("fidelity.unknown", 0.80) is None

    results = [
        MetricResult(id="fidelity.ks", value=0.78, details={}, family="fidelity"),  # 6 sd: warned
        MetricResult(id="fidelity.tvd", value=0.82, details={}, family="fidelity"),  # 4 sd: not
    ]
    out = Aggregator(calibration_bounds=bounds).aggregate(results, None)
    assert WARNING_Z == 5.0
    assert out["calibration_warnings"] == {"fidelity.ks": pytest.approx(6.0)}
    assert out["calibrated_metrics"] == {  # the warning does not touch the scores
        r.id: pytest.approx(bounds.normalize_with_bounds(r.id, r.value)) for r in results
    }

    quiet = [MetricResult(id="fidelity.ks", value=0.90, details={}, family="fidelity")]
    assert "calibration_warnings" not in Aggregator(calibration_bounds=bounds).aggregate(quiet, None)


def test_calibrated_scores_follow_the_anchors(calibrated):
    real, cfg_path, bounds = calibrated
    train, holdout, unseen = tercile_split(real, 200, seed=99)
    agg = Aggregator(calibration_bounds=bounds)
    for synth, expected in ((unseen, 1.0), (train.copy(), 0.0)):
        summary = Orchestrator().evaluate_dataframes(
            train,
            synth,
            {
                "data": {"schema": dict(SCHEMA), "target": "y"},
                "metrics": ["privacy.dcr", "privacy.mia"],
                "evaluation": {"n_runs": 1},
            },
            seed=0,
            holdout=holdout,
        )
        out = agg.aggregate(summary.results, None)
        assert out["privacy_score"] == pytest.approx(expected, abs=0.15)


def test_cli_calibrate_then_evaluate_uses_the_cached_bounds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = _table(300, 0)
    real.to_csv(tmp_path / "real.csv", index=False)
    cfg_path = _config_file(
        tmp_path,
        calibration={"n_iterations": 2, "cache_dir": str(tmp_path / "cache")},
    )
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    cfg["metrics"] = ["fidelity.ks", "privacy.dcr"]
    cfg["data"]["real"] = str(tmp_path / "real.csv")
    cfg["data"]["synthetic"] = str(tmp_path / "real.csv")
    Path(cfg_path).write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert cli.main(["calibrate", "-c", cfg_path]) == 0
    cached = list((tmp_path / "cache").glob("bounds_*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text(encoding="utf-8"))["format"] == 3

    # The table as the CLI read it (a CSV round trip moves floats by 1 ulp,
    # which changes the data fingerprint).
    real = pd.read_csv(tmp_path / "real.csv")
    calls = []
    monkeypatch.setattr(
        MetricCalibrator, "calibrate", lambda *a, **k: calls.append(1) or None
    )  # a cache miss would call it
    summary = Orchestrator().evaluate_dataframes(
        real, _table(300, 5), cfg, seed=0, config_path=cfg_path, holdout=_table(100, 9)
    )
    assert calls == []
    assert summary.aggregates["calibration_method"] == "per_metric"
    assert set(summary.aggregates["calibrated_metrics"]) <= {"fidelity.ks", "privacy.dcr"}
    assert math.isfinite(summary.aggregates["composite_score"])
