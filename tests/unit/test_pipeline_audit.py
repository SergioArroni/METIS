"""Pipeline, API and configuration findings of the 2026-10-08 audit (A2–A14).

Every test here failed with the code before the fix.  Only cheap metrics
(fidelity / ``privacy.dcr``) and a probe metric registered for the test are
used; utility metrics are only requested where they fail before training.
"""

from __future__ import annotations

import inspect
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metis.application.orchestrator import Orchestrator
from metis.application.pipeline import calibrator as calibrator_mod
from metis.application.pipeline.preprocessor import DataPreprocessor
from metis.domain.contracts import TypeSchema
from metis.domain.entities import MetricResult
from metis.domain.errors import ConfigError
from metis.infrastructure.io.loaders import load_csv
from metis.infrastructure.metrics.base import MetricBase
from metis.infrastructure.metrics.registry import get_metric_registry
from metis.infrastructure.preprocess.caster import SimpleCaster
from metis.infrastructure.runtime.config import (
    UnknownConfigKeyWarning,
    find_unknown_config_keys,
    load_config,
)
from metis.interface import cli
from metis.interface.sdk import Evaluator

PROBE_ID = "fidelity.audit_probe"


class _ProbeMetric(MetricBase):
    """Records the context and constructor arguments it receives."""

    family = "fidelity"
    requires_data = "both"
    seen: list[dict] = []

    def __init__(self, scale: float = 1.0, label: str = "x") -> None:
        super().__init__()
        self.scale = scale
        self.label = label

    def fit(self, real, synth, context):
        self._setup(real, synth, context)
        return self

    def compute(self) -> MetricResult:
        holdout = self._context.get("holdout")
        _ProbeMetric.seen.append(
            {
                "target": self._context.get("target"),
                "holdout": None if holdout is None else holdout.copy(),
                "real_columns": list(self._real_data.columns),
                "cat_columns": self._get_categorical_columns(),
                "num_columns": self._get_numeric_columns(),
                "scale": self.scale,
                "label": self.label,
                "seed": self._context.get("seed"),
            }
        )
        value = (self._context.get("seed", 0) % 10) / 10 * self.scale
        return MetricResult(id=PROBE_ID, value=value, details={}, family="fidelity")


@pytest.fixture
def probe():
    registry = get_metric_registry()
    registry._metrics[PROBE_ID] = _ProbeMetric
    _ProbeMetric.seen = []
    yield _ProbeMetric
    registry._metrics.pop(PROBE_ID, None)
    _ProbeMetric.seen = []


def _table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "c": rng.choice(["a", "b", "c"], size=n),
            "code": rng.choice([1, 2, 3, 4], size=n),
            "y": rng.choice(["yes", "no"], size=n),
        }
    )


SCHEMA = {"x": "continuous", "c": "nominal", "code": "nominal", "y": "nominal"}


def _cfg(metrics, **extra) -> dict:
    cfg = {"data": {"schema": dict(SCHEMA), "target": "y"}, "metrics": list(metrics)}
    for key, value in extra.items():
        cfg[key] = value
    return cfg


def _result(summary, metric_id):
    # some metrics report their short id ("tvd" for "fidelity.tvd")
    (r,) = [r for r in summary.results if r.id == metric_id or metric_id.endswith("." + r.id)]
    return r


# ── A6: 1 and 1.0 are the same category ──────────────────────────────────


def test_a6_int_real_and_float_synth_codes_are_the_same_category():
    real = _table(300, 0)
    synth = real.copy()
    synth["code"] = synth["code"].astype(float)  # a generator returning floats

    summary = Evaluator().evaluate(real, synth, _cfg(["fidelity.tvd"]))

    tvd = _result(summary, "fidelity.tvd")
    assert "error" not in tvd.details, tvd.details
    # identical data → the categorical TVD must be ~0 (it was 1 for ``code``)
    assert tvd.details["code"]["raw_value"] == pytest.approx(0.0, abs=1e-9)
    assert tvd.value == pytest.approx(1.0)
    caster = SimpleCaster(TypeSchema(columns={"code": "nominal"})).fit(real)
    real_cat, _ = caster.transform(real)
    synth_cat, _ = caster.transform(synth)
    assert set(synth_cat["code"]) == set(real_cat["code"]) == {"1", "2", "3", "4"}


def test_a6_float_booleans_and_codes_are_read_not_dropped():
    schema = TypeSchema(columns={"b": "nominal", "k": "nominal", "o": "nominal"})
    real = pd.DataFrame({"b": [1, 0, True], "k": [7, 8, 9], "o": [1, 2, 3]})
    synth = pd.DataFrame({"b": [1.0, 0.0, 1.0], "k": [7.0, 8.0, 9.0], "o": [1.0, 2.0, 3.5]})
    caster = SimpleCaster(schema).fit(real)

    real_cat, _ = caster.transform(real)
    synth_cat, _ = caster.transform(synth)

    # Booleans are nominal (Phase −1 types, 2026-10-19): read as their canonical
    # string, never as NaN (a float boolean used to drop the row).
    assert list(real_cat["b"]) == ["1", "0", "true"]
    assert list(synth_cat["b"]) == ["1", "0", "1"]
    assert list(synth_cat["k"]) == ["7", "8", "9"]  # was "7.0"
    assert list(synth_cat["o"]) == ["1", "2", "3.5"]


def test_a6_ordinal_levels_match_float_values():
    schema = TypeSchema(columns={"o": {"type": "ordinal", "levels": [1, 2, 3]}})
    caster = SimpleCaster(schema).fit(pd.DataFrame({"o": [1, 2, 3]}))
    _, num = caster.transform(pd.DataFrame({"o": [1.0, 2.0, 3.0]}))
    assert list(num["o__ord"]) == [0.0, 0.5, 1.0]


# ── A9: NaT is NaN, not -9.2e9 ───────────────────────────────────────────


def test_a9_datetime_nat_becomes_nan():
    schema = TypeSchema(columns={"d": {"type": "continuous", "datetime": True}})
    df = pd.DataFrame({"d": ["2020-01-01", "not a date", None]})
    _, num = SimpleCaster(schema).fit(df).transform(df)
    assert num["d__ts"].iloc[0] == pytest.approx(1577836800.0)
    assert math.isnan(num["d__ts"].iloc[1])
    assert math.isnan(num["d__ts"].iloc[2])


# ── A8: the CSV loader keeps NaN as NaN; CLI and SDK agree ───────────────


def test_a8_load_csv_keeps_missing_strings_as_nan(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,b\nx, 1\n,2\n  y ,3\n", encoding="utf-8")
    df = load_csv(str(path))
    assert df["a"].iloc[0] == "x"
    assert pd.isna(df["a"].iloc[1])  # was the string "nan"
    assert df["a"].iloc[2] == "y"
    assert "nan" not in set(df["a"].dropna())


def test_a8_file_pipeline_and_sdk_give_the_same_result(tmp_path):
    real = _table(200, 0)
    synth = _table(200, 1)
    real.loc[[3, 10, 20], "c"] = np.nan
    synth.loc[[5, 6], "c"] = np.nan
    real_path, synth_path = tmp_path / "real.csv", tmp_path / "synth.csv"
    real.to_csv(real_path, index=False)
    synth.to_csv(synth_path, index=False)
    cfg = _cfg(["fidelity.tvd", "fidelity.ks"])
    file_cfg = {
        **cfg,
        "data": {**cfg["data"], "real": str(real_path), "synthetic": str(synth_path)},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(file_cfg), encoding="utf-8")

    from_file = Orchestrator().run(str(cfg_path))
    from_sdk = Evaluator().evaluate(pd.read_csv(real_path), pd.read_csv(synth_path), cfg)

    # Rows with gaps are kept (the three NaN of ``c`` used to drop three rows).
    assert from_file.artifacts["rows"]["real"]["rows_out"] == 200
    assert from_file.artifacts["rows"]["real"]["cells_missing"] == 3
    assert from_file.artifacts["rows"] == from_sdk.artifacts["rows"]
    for metric_id in ("fidelity.tvd", "fidelity.ks"):
        assert _result(from_file, metric_id).value == pytest.approx(
            _result(from_sdk, metric_id).value
        )


# ── A7: invalid synthetic values are published (no row is dropped) ──────


def test_a7_invalid_synthetic_values_are_reported_not_dropped():
    real = _table(100, 0)
    synth = _table(100, 1)
    synth["x"] = synth["x"].astype(object)
    synth.loc[[0, 1, 2], "x"] = "abc"  # not a number in a continuous column → a gap, counted
    synth.loc[5, "x"] = np.nan  # a gap, kept
    cfg = _cfg(["fidelity.ks"])

    summary = Evaluator().evaluate(real, synth, cfg)

    art = summary.artifacts
    assert art["synth_rows_dropped"] == 0
    assert art["synth_cells_invalid"] == 3
    assert art["synth_rows_with_invalid"] == 3
    assert art["rows"]["synth"] == {
        "rows_in": 100,
        "rows_with_missing": 1,
        "cells_missing": 1,
        "rows_with_invalid": 3,
        "cells_invalid": 3,
        "invalid_by_column": {"x": 3},
        "rows_dropped_nan": 0,
        "rows_dropped_cast": 0,
        "rows_out": 100,
    }
    assert any("synthetic value(s)" in w for w in art["warnings"])


# ── A10: no target only blanks the utility metrics ───────────────────────


def test_a10_missing_target_only_affects_utility():
    cfg = _cfg(["fidelity.ks", "utility.tstr"])
    del cfg["data"]["target"]

    summary = Evaluator().evaluate(_table(80, 0), _table(80, 1), cfg)

    ks = _result(summary, "fidelity.ks")
    assert "error" not in ks.details and math.isfinite(ks.value)
    tstr = _result(summary, "utility.tstr")
    assert math.isnan(tstr.value)
    assert "data.target" in tstr.details["error"]
    assert any("data.target" in w for w in summary.artifacts["warnings"])


# ── A5: context["target"] is filled, and excluded from column helpers ────


def test_a5_context_target_is_the_first_target(probe):
    Evaluator().evaluate(_table(60, 0), _table(60, 1), _cfg([PROBE_ID]))
    (seen,) = probe.seen
    assert seen["target"] == "y"
    assert "y" not in seen["cat_columns"]
    assert "c" in seen["cat_columns"]

    cfg = _cfg([PROBE_ID])
    del cfg["data"]["target"]
    Evaluator().evaluate(_table(60, 0), _table(60, 1), cfg)
    assert probe.seen[-1]["target"] is None
    assert "y" in probe.seen[-1]["cat_columns"]


def test_low_datetime_target_is_found_after_renaming(probe):
    real, synth = _table(60, 0), _table(60, 1)
    for df, s in ((real, 0), (synth, 1)):
        df["when"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(
            np.random.default_rng(s).integers(0, 1000, len(df)), unit="D"
        )
    cfg = _cfg([PROBE_ID])
    cfg["data"]["schema"]["when"] = {"type": "continuous", "datetime": True}
    cfg["data"]["target"] = "when"

    Evaluator().evaluate(real, synth, cfg)  # raised "Target column 'when' not found"

    (seen,) = probe.seen
    assert "when__ts" in seen["real_columns"]
    assert "when__ts" not in seen["num_columns"]


# ── A2: metric_params reach the constructor, validated by signature ──────


def test_a2_metric_params_reach_the_constructor(probe):
    cfg = _cfg([PROBE_ID], evaluation={"metric_params": {PROBE_ID: {"scale": 2.0, "label": "z"}}})
    Evaluator().evaluate(_table(60, 0), _table(60, 1), cfg)
    assert probe.seen[-1]["scale"] == 2.0
    assert probe.seen[-1]["label"] == "z"


def test_a2_unknown_metric_param_is_a_clear_metric_error(probe):
    cfg = _cfg(
        [PROBE_ID, "fidelity.ks"],
        evaluation={"metric_params": {PROBE_ID: {"scael": 2.0}}},
    )
    summary = Evaluator().evaluate(_table(60, 0), _table(60, 1), cfg)
    probe_result = _result(summary, PROBE_ID)
    assert math.isnan(probe_result.value)
    assert "scael" in probe_result.details["error"]
    assert "scale" in probe_result.details["error"]  # lists the accepted ones
    assert math.isfinite(_result(summary, "fidelity.ks").value)


def test_a2_real_metric_params(monkeypatch):
    dcr_cls = get_metric_registry().get("privacy.dcr")
    received = []
    original_init = dcr_cls.__init__

    def spy(self, *args, **kwargs):
        received.append(kwargs)
        original_init(self, *args, **kwargs)

    spy.__signature__ = inspect.signature(original_init)
    monkeypatch.setattr(dcr_cls, "__init__", spy)
    cfg = _cfg(
        ["privacy.dcr"],
        evaluation={"metric_params": {"privacy.dcr": {"columns": ["x", "c"]}}},
    )
    summary = Evaluator().evaluate(_table(80, 0), _table(80, 1), cfg)
    assert "error" not in _result(summary, "privacy.dcr").details
    assert received == [{"columns": ["x", "c"]}]


def test_a2_metric_params_for_unknown_metric_id_is_a_config_error():
    cfg = _cfg(["fidelity.ks"], evaluation={"metric_params": {"fidelity.nope": {"a": 1}}})
    with pytest.raises(ConfigError, match="fidelity.nope"):
        Evaluator().evaluate(_table(60, 0), _table(60, 1), cfg)


# ── A4: unknown config keys are reported ─────────────────────────────────


def test_a4_sdk_warns_on_unknown_keys():
    cfg = _cfg(["fidelity.ks"], evalaution={"n_runs": 1})
    cfg["data"]["tagret"] = "y"
    with pytest.warns(UnknownConfigKeyWarning, match="evalaution") as record:
        summary = Evaluator().evaluate(_table(60, 0), _table(60, 1), cfg)
    assert "data.tagret" in str(record[0].message)
    assert any("evalaution" in w for w in summary.artifacts["warnings"])


def test_a4_sdk_strict_mode_raises_and_structure_is_checked():
    cfg = _cfg(["fidelity.ks"], evalaution={"n_runs": 1})
    with pytest.raises(ConfigError, match="evalaution"):
        Evaluator(strict_config=True).evaluate(_table(60, 0), _table(60, 1), cfg)
    with pytest.raises(ConfigError, match="schema"):
        Evaluator().evaluate(_table(60, 0), _table(60, 1), {"data": {}, "metrics": ["x"]})


def test_a4_load_config_warns_on_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"real": "r.csv", "synthetic": "s.csv", "schema": {"a": "continuous"}},
                "metrics": ["fidelity.ks"],
                "calibration": {"n_iteration": 3},
            }
        ),
        encoding="utf-8",
    )
    with pytest.warns(UnknownConfigKeyWarning, match="calibration.n_iteration"):
        load_config(str(path))


def test_a4_shipped_configs_have_no_unknown_keys():
    for path in Path("metis/configs").glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert find_unknown_config_keys(cfg) == [], path


# ── Holdout contract ─────────────────────────────────────────────────────


def test_holdout_reaches_the_context_preprocessed_like_the_real(probe):
    real, synth = _table(80, 0), _table(80, 1)
    holdout = _table(30, 2)
    holdout["code"] = holdout["code"].astype(float)
    holdout["id"] = range(30)
    holdout.loc[0, "x"] = np.nan

    cfg = _cfg([PROBE_ID])
    summary = Evaluator().evaluate(real, synth, cfg, holdout=holdout)

    seen = probe.seen[-1]
    hold = seen["holdout"]
    assert isinstance(hold, pd.DataFrame)
    assert list(hold.columns) == seen["real_columns"]  # same columns, extra one dropped
    assert len(hold) == 30  # the row with a gap is kept, like in the real table
    assert hold["x"].isna().sum() == 1
    assert set(hold["code"]) <= {"1", "2", "3", "4"}  # cast with the real caster
    assert summary.artifacts["holdout_rows"] == 30

    Evaluator().evaluate(real, synth, cfg)
    assert probe.seen[-1]["holdout"] is None


def test_holdout_from_file_config(tmp_path, probe):
    paths = {}
    for name, seed, n in (("real", 0, 80), ("synth", 1, 80), ("holdout", 2, 25)):
        paths[name] = tmp_path / f"{name}.csv"
        _table(n, seed).to_csv(paths[name], index=False)
    cfg = _cfg([PROBE_ID])
    cfg["data"].update(
        real=str(paths["real"]), synthetic=str(paths["synth"]), holdout=str(paths["holdout"])
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    summary = Orchestrator().run(str(cfg_path))

    assert len(probe.seen[-1]["holdout"]) == 25
    assert summary.artifacts["holdout_rows"] == 25


def test_holdout_with_missing_schema_column_fails_clearly():
    holdout = _table(30, 2).drop(columns=["c"])
    with pytest.raises(Exception, match="missing in holdout data"):
        Evaluator().evaluate(_table(60, 0), _table(60, 1), _cfg(["fidelity.ks"]), holdout=holdout)


# ── A13: the real preprocessing is reused, results unchanged ─────────────


def test_a13_reference_is_cached_and_results_are_identical(monkeypatch):
    real = _table(200, 0)
    cfg = _cfg(["fidelity.tvd", "fidelity.ks", "privacy.dcr"])
    calls = {"n": 0}
    original = DataPreprocessor.prepare_reference

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataPreprocessor, "prepare_reference", counting)

    cached, plain = Evaluator(), Evaluator(cache_reference=False)
    for seed in (1, 2, 3):
        synth = _table(200, seed)
        # privacy.dcr needs non-members (2026-10-14)
        a = cached.evaluate(real, synth, cfg, holdout=_table(200, 9))
        b = plain.evaluate(real, synth, cfg, holdout=_table(200, 9))
        assert [r.value for r in a.results] == pytest.approx([r.value for r in b.results])
        assert a.aggregates["composite_score"] == pytest.approx(b.aggregates["composite_score"])
    assert calls["n"] == 1 + 3  # once for the cached evaluator, every call for the other

    # a different data config is a different reference
    cfg2 = _cfg(["fidelity.ks"])
    cfg2["data"]["schema"]["code"] = {"type": "discrete", "levels": [1, 2, 3, 4]}
    cached.evaluate(real, _table(200, 4), cfg2)
    assert calls["n"] == 5
    # mutating the real table in place invalidates the key
    real.loc[0, "x"] = 99.0
    cached.evaluate(real, _table(200, 4), cfg)
    assert calls["n"] == 6


# ── A14: CLI calibrate and the pipeline share data, defaults and seed ────


def test_a14_cli_calibrate_matches_the_pipeline(tmp_path, monkeypatch):
    real = _table(120, 0)
    real["id"] = range(120)
    real_path = tmp_path / "mydata.csv"
    real.to_csv(real_path, sep=";", index=False)
    cfg = _cfg(["fidelity.ks"])
    cfg["data"].update(real=str(real_path), synthetic="None", real_separator=";")
    cfg["data"]["schema"]["id"] = "id"
    cfg["reproducibility"] = {"seed": 7}
    cfg["calibration"] = {"n_iterations": 3}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    calls = []

    class _FakeBounds:
        def get_summary(self):
            return "ok"

    class _FakeManager:
        def __init__(self, *args, **kwargs):
            calls.append({"init": kwargs})

        def get_or_calibrate(self, real_data, **kwargs):
            calls[-1].update(kwargs=kwargs, data=real_data.copy())
            return _FakeBounds()

    monkeypatch.setattr(calibrator_mod, "CacheManager", _FakeManager)

    assert cli.main(["calibrate", "-c", str(cfg_path)]) == 0
    ref = DataPreprocessor().prepare_reference(
        pd.read_csv(real_path, sep=";"), load_config(str(cfg_path))
    )
    calibrator_mod.CalibrationStep().calibrate_if_needed(
        ref.real_clean, load_config(str(cfg_path)), str(cfg_path)
    )

    cli_call, pipe_call = calls
    assert cli_call["init"] == pipe_call["init"] == {"dataset_name": "mydata"}
    assert cli_call["kwargs"] == pipe_call["kwargs"]
    assert cli_call["kwargs"]["base_seed"] == 7
    assert cli_call["kwargs"]["sample_percentage"] == 100.0
    assert cli_call["kwargs"]["n_iterations"] == 3
    pd.testing.assert_frame_equal(cli_call["data"], pipe_call["data"])
    assert "id" not in cli_call["data"].columns


# ── Low: the multi-run summary describes every run ───────────────────────


def test_low_multi_run_results_are_the_mean_over_runs(tmp_path, probe):
    paths = {}
    for name, seed in (("real", 0), ("synth", 1)):
        paths[name] = tmp_path / f"{name}.csv"
        _table(60, seed).to_csv(paths[name], index=False)
    cfg = _cfg([PROBE_ID])
    cfg["data"].update(real=str(paths["real"]), synthetic=str(paths["synth"]))
    cfg["evaluation"] = {"n_runs": 3}
    cfg["reproducibility"] = {"seed": 1}
    cfg["report"] = {"formats": ["json"], "output_dir": str(tmp_path / "rep")}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = Orchestrator().run(str(cfg_path))

    per_run = [s["seed"] % 10 / 10 for s in probe.seen]
    assert len(per_run) == 3 and len(set(per_run)) > 1
    result = _result(summary, PROBE_ID)
    assert result.value == pytest.approx(np.mean(per_run))
    assert result.details["multi_run"]["values"] == pytest.approx(per_run)
    assert summary.artifacts["summary_mode"] == "mean_of_runs"
