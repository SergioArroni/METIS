"""The benchmark trains on one part of the real table and evaluates with a holdout (M1, M5)."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metis.sota_models.comparison.orchestrator import BenchmarkOrchestrator


def _write(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 300
    real = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.gamma(2.0, 1.0, size=n),
            "c": rng.choice(["a", "b"], size=n),
            "lvl": rng.choice([1, 2, 3], size=n),
            "y": rng.choice(["yes", "no"], size=n),
        }
    )
    real.to_csv(tmp_path / "real.csv", index=False)
    cfg = {
        "data": {
            "real": str(tmp_path / "real.csv"),
            "target": "y",
            "task_type": "classification",
            "schema": {
                "x0": "continuous",
                "x1": "continuous",
                "c": "nominal",
                "lvl": {"type": "ordinal", "levels": [1, 2, 3]},
                "y": "nominal",
            },
        },
        "metrics": ["fidelity.marginal.coverage", "privacy.dcr", "privacy.mia"],
        "reproducibility": {"seed": 7},
        "calibration": {"enabled": False},
        "benchmark": {
            "output_dir": str(tmp_path / "out"),
            "n_runs": 2,
            "generators": [
                {"name": "real_data", "params": {}},
                {"name": "bootstrap", "params": {}},
                {"name": "delete_zero", "params": {"deletion_rate": 0.5}},
            ],
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_generators_are_trained_on_a_part_and_evaluated_with_a_holdout(tmp_path):
    orch = BenchmarkOrchestrator(str(_write(tmp_path)))
    results = orch.run_benchmark()

    for gen in ("real_data", "bootstrap", "delete_zero"):
        for run in results[gen].values():
            assert "error" not in run, run
            # 60/20/20 of 300 rows (thirds until 2026-10-19)
            assert run["n_train"] == 180 and run["n_holdout"] == 60
            summary = run["run_summary"]
            # The holdout reaches METIS: MIA applies (it was NaN without it).
            (mia,) = [r for r in summary.results if r.id == "privacy.mia"]
            assert mia.status == "ok"
            assert summary.artifacts["synth_rows_dropped"] == 0
            assert Path(run["synthetic_path"]).exists()  # artifacts kept by default

    def priv(gen):
        return np.mean(
            [r["run_summary"].aggregates["privacy_score"] for r in results[gen].values()]
        )

    # Unseen real rows are the best case of privacy; a bootstrap of the
    # training part is a copy.
    assert priv("real_data") > 0.9
    assert priv("bootstrap") < 0.3
