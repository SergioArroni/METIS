"""Second review of the pipeline, calibration, I/O and benchmark (R-A*).

Every test here failed with the code before its fix.  Only cheap metrics
(``fidelity.ks`` / ``fidelity.tvd``) and fakes are used: no generator is
trained and no utility metric runs.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metis.application.orchestrator import Orchestrator
from metis.application.pipeline import calibrator as calibrator_mod
from metis.application.pipeline.loader import DataLoader
from metis.application.pipeline.preprocessor import DataPreprocessor
from metis.application.pipeline.reporter import ReportGenerator
from metis.calibrate.cache.cache_manager import CacheManager
from metis.calibrate.cache.fingerprint import compute_config_fingerprint
from metis.calibrate.core.bounds import CalibrationBounds
from metis.calibrate.core.calibrator import MetricCalibrator
from metis.calibrate.strategies.base import BaseCalibrationStrategy
from metis.calibrate.utils.evaluator import InMemoryEvaluator
from metis.domain.contracts import TypeSchema
from metis.domain.entities import EvalPlan, MetricResult, RunSummary
from metis.domain.errors import ConfigError, METISError
from metis.infrastructure.preprocess.caster import SimpleCaster
from metis.interface import cli
from metis.interface.sdk import Evaluator
from metis.sota_models.comparison.statistical_tests import FriedmanNemenyiTest

REPO = Path(__file__).resolve().parents[2]


def _table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "c": rng.choice(["a", "b", "c"], size=n),
            "y": rng.choice(["yes", "no"], size=n),
        }
    )


SCHEMA = {"x": "continuous", "c": "nominal", "y": "nominal"}


def _cfg(metrics, **extra) -> dict:
    cfg = {"data": {"schema": dict(SCHEMA), "target": "y"}, "metrics": list(metrics)}
    cfg.update(extra)
    return cfg


# ── R-A1: importing METIS must not reconfigure the host's logging ────────


def test_ra1_import_leaves_root_logger_and_warning_filters_alone():
    code = (
        "import logging, warnings\n"
        "root = logging.getLogger()\n"
        "h = logging.NullHandler(); root.addHandler(h); root.setLevel(logging.ERROR)\n"
        "before = list(warnings.filters)\n"
        "import metis.interface.sdk\n"
        "assert root.handlers == [h], root.handlers\n"
        "assert root.level == logging.ERROR, root.level\n"
        "new = [f for f in warnings.filters if f not in before]\n"
        "silenced = (FutureWarning, DeprecationWarning, UserWarning)\n"
        "bad = [f for f in new if f[0] == 'ignore' and f[2] in silenced]\n"
        "assert not bad, bad\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_ra1_cli_still_configures_logging(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: called.append(True))
    cli.main(["version"])
    assert called


# ── R-A2: calibration gets the raw (cleaned) real table, not a re-cast one ─


def _raw_mixed(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _table(n, seed)
    df["age"] = rng.integers(0, 90, n)
    df["when"] = (
        pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    ).astype(str)
    df["pid"] = range(n)
    return df


MIXED_SCHEMA = {
    **SCHEMA,
    "age": {"type": "discrete", "step": 1},
    "when": {"type": "continuous", "datetime": True},
    "pid": "id",
}


def test_ra2_calibration_receives_the_cleaned_raw_table(tmp_path, monkeypatch):
    real = _raw_mixed(80, 0)
    real.loc[3, "x"] = np.nan
    cfg = _cfg(["fidelity.ks"])
    cfg["data"]["schema"] = dict(MIXED_SCHEMA)
    cfg["calibration"] = {"n_iterations": 2}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    seen = []

    class _FakeManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_or_calibrate(self, real_data, **kwargs):
            seen.append(real_data.copy())
            return

    monkeypatch.setattr(calibrator_mod, "CacheManager", _FakeManager)
    Orchestrator().evaluate_dataframes(real, _raw_mixed(80, 1), cfg, config_path=str(cfg_path))

    (data,) = seen
    # Same columns and values a synthetic table has before preprocessing, so
    # the calibration evaluations preprocess them exactly once.
    assert set(data.columns) == {"x", "c", "y", "age", "when"}
    assert len(data) == 80  # the row with a gap is kept
    expected = real.drop(columns="pid").reset_index(drop=True)
    pd.testing.assert_frame_equal(data, expected)


def test_ra2_calibration_evaluation_equals_a_normal_evaluation(tmp_path):
    cfg = _cfg(["fidelity.ks"])
    cfg["data"]["schema"] = dict(MIXED_SCHEMA)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    raw = _raw_mixed(80, 0)
    ref = DataPreprocessor().prepare_reference(raw, cfg)
    half_a, half_b = ref.real_clean.iloc[:40], ref.real_clean.iloc[40:]
    _, values, _ = InMemoryEvaluator(str(cfg_path)).evaluate(half_a, half_b, seed=0)

    normal = Evaluator().evaluate(raw.iloc[:40], raw.iloc[40:], cfg, seed=0)
    (ks,) = normal.results
    # the datetime and discrete columns take part (before: ``when`` was
    # dropped and ``age`` re-binned from its already-binned values)
    assert {"age", "when__ts", "x"} <= set(ks.details)
    assert list(values["fidelity"].values()) == pytest.approx([ks.value])


# ── R-A3: a failed calibration strategy is an error, not empty bounds ────


class _OkStrategy(BaseCalibrationStrategy):
    def __init__(self, score):
        super().__init__()
        self.score = score

    def get_strategy_name(self):
        return f"ok{self.score}"

    def _run_single_iteration(self, real_data, config_template_path, sample_size, seed):
        return {"fidelity": self.score}, {"fidelity": {"m": self.score}}, {"fidelity": {"m": 0}}


class _FailingStrategy(_OkStrategy):
    def _run_single_iteration(self, real_data, config_template_path, sample_size, seed):
        raise ValueError("boom")


def test_ra3_strategy_with_no_successful_iteration_raises():
    with pytest.raises(RuntimeError, match="boom"):
        _FailingStrategy(0.5).calibrate(pd.DataFrame({"a": [1]}), "cfg", 2, 1, 0)


def test_ra3_calibrator_does_not_return_bounds_without_families():
    cal = MetricCalibrator(upper_strategy=_FailingStrategy(1.0), lower_strategy=_OkStrategy(0.1))
    with pytest.raises(RuntimeError):
        cal.calibrate(pd.DataFrame({"a": range(4)}), "cfg", n_iterations=2, sample_size=4)

    ok = MetricCalibrator(upper_strategy=_OkStrategy(0.9), lower_strategy=_OkStrategy(0.1))
    bounds = ok.calibrate(
        pd.DataFrame({"a": range(4)}), "cfg", n_iterations=2, sample_size=4, tune_aggregators=False
    )
    assert bounds.get_bounds("m") == (0.1, 0.9)  # per metric now


# ── R-A4: the calibration cache key covers what changes the bounds ───────


def _fp(tmp_path, cfg) -> str:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return compute_config_fingerprint(str(p))


def test_ra4_config_fingerprint_depends_on_target_params_and_weights(tmp_path):
    base = _cfg(["fidelity.ks"])
    fp0 = _fp(tmp_path, base)
    assert _fp(tmp_path, {**base, "data": {**base["data"], "target": "c"}}) != fp0
    assert _fp(tmp_path, {**base, "evaluation": {"metric_params": {"fidelity.ks": {}}}}) == fp0
    assert (
        _fp(
            tmp_path,
            {**base, "evaluation": {"metric_params": {"privacy.dcr": {"percentile_threshold": 1}}}},
        )
        != fp0
    )
    assert _fp(tmp_path, {**base, "evaluation": {"ml_efficiency_config": {"n_trials": 3}}}) != fp0
    assert _fp(tmp_path, {**base, "aggregation": {"weights": {"utility": 0}}}) != fp0
    assert _fp(tmp_path, {**base, "aggregation": {"risk_aversion": 3.0}}) != fp0
    # keys that do not change the bounds leave the key alone
    assert _fp(tmp_path, {**base, "report": {"formats": [], "output_dir": "x"}}) == fp0


def test_ra4_tune_aggregators_no_longer_invalidates_the_cache(tmp_path, monkeypatch):
    """Calibration is per metric and ignores ``tune_aggregators``: one calibration serves both."""
    cfg = _cfg(["fidelity.ks"])
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    runs = []

    class _FakeCalibrator:
        def __init__(self, *a, **k):
            pass

        def calibrate(self, tune_aggregators, **kwargs):
            runs.append(tune_aggregators)
            b = CalibrationBounds()
            b.set_bounds("fidelity.ks", 0.1, 0.9)
            return b

    from metis.calibrate.cache import cache_manager as cm

    monkeypatch.setattr(cm, "MetricCalibrator", _FakeCalibrator)
    mgr = CacheManager(cache_dir=str(tmp_path / "cache"))
    real = _table(30, 0)
    mgr.get_or_calibrate(real, str(cfg_path), n_iterations=2, tune_aggregators=True)
    mgr.get_or_calibrate(real, str(cfg_path), n_iterations=2, tune_aggregators=True)
    mgr.get_or_calibrate(real, str(cfg_path), n_iterations=2, tune_aggregators=False)
    assert runs == [True]


def test_ra4_invalidate_cache_finds_files_from_other_days(tmp_path):
    mgr = CacheManager(cache_dir=str(tmp_path))
    old = mgr._get_cache_path("calibration_k", date_tag="20200101")
    old.write_text("{}", encoding="utf-8")
    mgr.invalidate_cache("calibration_k")
    assert not old.exists()


def test_ra4_calibration_cache_dir_is_configurable(tmp_path, monkeypatch):
    inits = []

    class _FakeManager:
        def __init__(self, *args, **kwargs):
            inits.append(kwargs)

        def get_or_calibrate(self, real_data, **kwargs):
            return None

    monkeypatch.setattr(calibrator_mod, "CacheManager", _FakeManager)
    cfg = _cfg(["fidelity.ks"], calibration={"cache_dir": str(tmp_path / "cc")})
    calibrator_mod.CalibrationStep().run_calibration(_table(20, 0), cfg, "cfg.yaml")
    assert inits[-1]["cache_dir"] == str(tmp_path / "cc")


# ── R-A5: the file pipeline honours evaluation.seed like the SDK ─────────


def test_ra5_file_pipeline_uses_evaluation_seed(tmp_path, monkeypatch):
    paths = {}
    for name, seed in (("real", 0), ("synth", 1)):
        paths[name] = tmp_path / f"{name}.csv"
        _table(40, seed).to_csv(paths[name], index=False)
    cfg = _cfg(["fidelity.ks"], evaluation={"seed": 5}, reproducibility={"seed": 1})
    cfg["data"].update(real=str(paths["real"]), synthetic=str(paths["synth"]))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    summary = Orchestrator().run(str(cfg_path))
    sdk = Evaluator().evaluate(_table(40, 0), _table(40, 1), cfg)
    assert summary.artifacts["seed"] == sdk.artifacts["seed"] == 5


# ── R-A6 / R-A7: loader honours its documented separator order and Parquet ─


def test_ra6_holdout_separator_falls_back_to_the_real_one():
    data_cfg = {"real_separator": ";", "separator": ","}
    params = DataLoader._holdout_csv_params(data_cfg)
    assert params == {"sep": ";"}
    assert DataLoader._holdout_csv_params({"separator": "|"}) == {"sep": "|"}
    assert DataLoader._holdout_csv_params({"holdout_separator": "\t", "real_separator": ";"}) == {
        "sep": "\t"
    }


def test_ra7_parquet_paths_are_read_as_parquet(tmp_path, monkeypatch):
    from metis.infrastructure.io import loaders

    target = tmp_path / "real.parquet"
    target.write_bytes(b"PAR1")
    frame = _table(12, 0)
    calls = []

    def _fake_read_parquet(path, **kwargs):
        calls.append(path)
        return frame

    monkeypatch.setattr(loaders.pd, "read_parquet", _fake_read_parquet)
    out = loaders.load_table(str(target), sep=";")
    assert calls and out.equals(frame)
    cfg = {"data": {"real": str(target), "real_separator": ";"}}
    assert DataLoader().load_real(cfg).equals(frame)


# ── R-A8 / R-A9 / R-A10: caster ──────────────────────────────────────────


def test_ra8_id_type_is_case_insensitive_and_reported():
    real = _table(30, 0)
    real["PID"] = range(30)
    cfg = _cfg(["fidelity.ks"])
    cfg["data"]["schema"]["PID"] = "ID"
    summary = Evaluator().evaluate(real, real.copy(), cfg)
    assert "PID" not in summary.artifacts["schema_applied"]
    assert summary.artifacts["excluded_id_columns"] == ["PID"]


def test_ra9_geospatial_numeric_columns_are_numeric_without_lat_lon():
    df = pd.DataFrame({"latitude": [41.38, 41.39, 41.40], "zone": ["n", "s", "n"]})
    caster = SimpleCaster(TypeSchema(columns={"latitude": "continuous", "zone": "nominal"}))
    cat, num = caster.fit(df).transform(df)
    assert list(num.columns) == ["latitude"]
    assert num["latitude"].tolist() == [41.38, 41.39, 41.40]
    assert list(cat.columns) == ["zone"]

    # Airbnb: latitude/longitude as text, and an undeclared lat/lon pair that
    # used to replace every geospatial column
    airbnb = pd.DataFrame(
        {
            "latitude": ["41,38", "41,39", "41,40"],
            "longitude": [2.17, 2.18, 2.19],
            "lat": [0.0, 0.0, 0.0],
            "lon": [1.0, 1.0, 1.0],
        }
    )
    schema = TypeSchema(columns={"latitude": "continuous", "longitude": "continuous"})
    cat, num = SimpleCaster(schema).fit(airbnb).transform(airbnb)
    assert cat.empty
    assert list(num.columns) == ["latitude", "longitude"]
    assert num["latitude"].tolist() == [41.38, 41.39, 41.40]


def test_ra10_ordinal_level_warning_uses_normalised_values():
    df = pd.DataFrame({"lvl": [1, 2, 3, 1]})
    caster = SimpleCaster(
        TypeSchema(columns={"lvl": {"type": "ordinal", "levels": ["1", "2", " 3 "]}})
    )
    _, warns = caster.fit(df).transform_to_entity(df, "real")
    assert warns == []
    _, warns = caster.transform_to_entity(df[df["lvl"] < 3], "synth")
    assert len(warns) == 1 and "3" in warns[0]


# ── R-A11: unknown metric ids fail the plan instead of becoming NaN ──────


def test_ra11_unknown_metric_id_is_a_config_error():
    with pytest.raises(ConfigError, match="fidelity.tvdd"):
        Evaluator().evaluate(_table(30, 0), _table(30, 1), _cfg(["fidelity.tvdd"]))


# ── R-A12: Friedman with two generators does not crash the report ────────


def test_ra12_friedman_with_two_methods_returns_a_clear_result():
    scores = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4, 0.5], "b": [0.5, 0.6, 0.7, 0.8, 0.9]})
    result = FriedmanNemenyiTest().compare(scores)
    assert math.isnan(result.p_value)
    assert "3" in result.interpretation


# ── R-A13 / R-A14: benchmark ─────────────────────────────────────────────


def test_ra13_benchmark_reads_the_global_separator(tmp_path):
    from metis.sota_models.comparison.orchestrator import BenchmarkOrchestrator

    cfg = _cfg(["fidelity.ks"])
    cfg["data"].update(real="x.csv", separator=";")
    cfg["benchmark"] = {"generators": [], "output_dir": str(tmp_path)}
    path = tmp_path / "b.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    view = BenchmarkOrchestrator(str(path)).benchmark_config
    assert view["real_data_separator"] == ";"
    assert view["synthetic_data_separator"] == ";"


def _summary(aggregates: dict) -> RunSummary:
    return RunSummary(plan=EvalPlan(metric_ids=["fidelity.ks"]), results=[], aggregates=aggregates)


def test_ra14_missing_family_scores_are_nan_not_zero(tmp_path):
    from metis.sota_models.comparison.orchestrator import BenchmarkOrchestrator
    from metis.sota_models.comparison.reporter import BenchmarkReporter

    results = {"g": {1: {"run_summary": _summary({"fidelity_score": 0.8, "composite_score": 0.8})}}}
    scores = BenchmarkReporter(results, str(tmp_path))._extract_scores()
    assert scores.loc[0, "fidelity"] == 0.8
    assert math.isnan(scores.loc[0, "utility"])

    bench = BenchmarkOrchestrator.__new__(BenchmarkOrchestrator)
    bench.results = results
    table = bench.get_aggregated_scores()
    assert math.isnan(table.loc[0, "utility_mean"])
    bench._save_results(tmp_path / "r.json")
    saved = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert '"utility": null' in saved


# ── R-A15: the BN generator reverses its discretisation reproducibly ─────


def test_ra15_bayesian_network_continuous_columns_are_seeded():
    from metis.sota_models.generators.bayesian_network import BayesianNetworkGenerator

    class _Sampler:
        def forward_sample(self, size, seed):
            return pd.DataFrame({"x": [0, 1, 2] * (size // 3)})

    def _make():
        gen = BayesianNetworkGenerator(random_state=3)
        gen._sampler = _Sampler()
        gen._is_fitted = True
        gen._constant_columns = {}
        gen._continuous_columns = ["x"]
        gen._discretize_info = {"x": [0.0, 1.0, 2.0, 3.0]}
        # Observed values of each bin: a bin is turned back into one of them.
        gen._bin_values = {
            "x": {0: np.array([0.1, 0.7]), 1: np.array([1.5]), 2: np.array([2.2, 2.9])}
        }
        gen._columns = ["x"]
        return gen

    np.random.seed(0)
    a = _make().generate(9)
    np.random.seed(1)
    b = _make().generate(9)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["x"]) <= {0.1, 0.7, 1.5, 2.2, 2.9}


# ── R-A16: bounds normalisation never maps NaN to 0 ──────────────────────


def test_ra16_normalize_with_bounds_keeps_nan_and_refuses_degenerate_bounds():
    b = CalibrationBounds()
    b.set_bounds("fidelity.ks", 0.2, 0.8)
    assert math.isnan(b.normalize_with_bounds("fidelity.ks", float("nan")))
    assert b.normalize_with_bounds("fidelity.ks", 0.5) == pytest.approx(0.5)
    b.set_bounds("privacy.dcr", 0.5, 0.5)  # no range: excluded, not a 0/1 step
    assert not b.is_calibratable("privacy.dcr")
    with pytest.raises(ValueError, match="not above"):
        b.normalize_with_bounds("privacy.dcr", 0.9)


# ── R-A17: report failures are not swallowed ─────────────────────────────


def test_ra17_report_failure_is_raised(tmp_path):
    summary = _summary({"composite_score": 0.5})
    cfg = {"report": {"formats": ["json", "pdf"], "output_dir": str(tmp_path)}}
    with pytest.raises(METISError, match="pdf"):
        ReportGenerator().generate(summary, cfg)
    assert (tmp_path / "summary.json").exists()  # the other formats still ran


# ── R-A18: calibration ignores metric values that are not finite ─────────


def test_ra18_in_memory_evaluator_skips_nan_metric_values(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(["fidelity.ks"])), encoding="utf-8")
    results = [
        MetricResult(id="privacy.mia", value=float("nan"), details={}, family="privacy"),
        MetricResult(id="privacy.dcr", value=0.7, details={}, family="privacy"),
    ]

    class _FakeOrch:
        def evaluate_dataframes(self, **kwargs):
            return RunSummary(
                plan=EvalPlan(metric_ids=["privacy.dcr"]),
                results=results,
                aggregates={"privacy_score": 0.7},
            )

    import metis.application.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "Orchestrator", _FakeOrch)
    _, values, raws = InMemoryEvaluator(str(cfg_path)).evaluate(_table(10, 0), _table(10, 1), 0)
    assert values == {"privacy": {"privacy.dcr": 0.7}}
    assert raws == {"privacy": {"privacy.dcr": 0.7}}


# ── Benchmark: generators train on the uncast real table, with its types ──


def test_benchmark_generators_train_on_the_uncast_table_with_types(tmp_path):
    from metis.sota_models.comparison.orchestrator import BenchmarkOrchestrator
    from metis.sota_models.generators.base import BaseGenerator
    from metis.sota_models.generators.registry import GeneratorRegistry

    seen: dict = {}

    class _Probe(BaseGenerator):
        def __init__(self, random_state=None, **kwargs):
            super().__init__(name="probe")
            self.random_state = random_state

        def fit(
            self, real_data, categorical_columns=None, ordinal_columns=None, continuous_columns=None
        ):
            seen.update(
                data=real_data.copy(),
                cat=categorical_columns,
                ord=ordinal_columns,
                num=continuous_columns,
            )
            self._real = real_data

        def generate(self, n_samples):
            return self._real.sample(n=n_samples, replace=True, random_state=self.random_state)

    rng = np.random.default_rng(0)
    real = _table(60, 0)
    real["size"] = rng.choice(["S", "M", "L"], 60)
    real["when"] = (
        pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 300, 60), unit="D")
    ).astype(str)
    real["pid"] = range(60)
    real_path = tmp_path / "real.csv"
    real.to_csv(real_path, index=False)
    cfg = _cfg(["fidelity.ks", "fidelity.tvd"])
    cfg["data"]["schema"].update(
        size={"type": "ordinal", "levels": ["S", "M", "L"]},
        when={"type": "continuous", "datetime": True},
        pid="id",
    )
    cfg["data"]["real"] = str(real_path)
    cfg["benchmark"] = {
        "generators": [{"name": "review_probe"}],
        "output_dir": str(tmp_path / "out"),
        "n_runs": 1,
    }
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    GeneratorRegistry.register("review_probe", _Probe)
    try:
        results = BenchmarkOrchestrator(str(cfg_path)).run_benchmark()
    finally:
        GeneratorRegistry.unregister("review_probe")

    data = seen["data"]
    assert set(data.columns) == {"x", "c", "y", "size", "when"}  # no size__ord / when__ts
    assert seen["ord"] == {"size": ["S", "M", "L"]}
    assert set(seen["cat"]) == {"c", "y"}
    assert set(seen["num"]) == {"x", "when"}
    assert pd.api.types.is_float_dtype(data["when"])  # seconds since the epoch

    (run,) = results["review_probe"].values()
    assert "error" not in run, run
    ids = {r.id for r in run["run_summary"].results}
    assert all("error" not in r.details for r in run["run_summary"].results), ids
    # the evaluation used the user's schema: datetime cast as usual
    assert "when__ts" in run["run_summary"].artifacts["schema_applied"] or any(
        "when__ts" in (r.details or {}) for r in run["run_summary"].results
    )
