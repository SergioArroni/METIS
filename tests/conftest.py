"""Shared fixtures for the METIS test suite."""

import numpy as np
import pandas as pd
import pytest

from metis.domain.entities import EvalPlan, MetricResult, RunSummary


@pytest.fixture(scope="session")
def sample_real_df() -> pd.DataFrame:
    """Small deterministic DataFrame mimicking a real dataset (20 rows, 5 cols)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=20).astype(float),
            "income": rng.uniform(20000, 120000, size=20),
            "score": rng.uniform(0, 1, size=20),
            "gender": rng.choice(["M", "F"], size=20),
            "city": rng.choice(["Madrid", "Barcelona", "Valencia", "Sevilla"], size=20),
        }
    )


@pytest.fixture(scope="session")
def sample_synth_df() -> pd.DataFrame:
    """Matching synthetic DataFrame with slight distribution shift."""
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "age": rng.integers(20, 75, size=20).astype(float),
            "income": rng.uniform(25000, 110000, size=20),
            "score": rng.uniform(0.1, 0.9, size=20),
            "gender": rng.choice(["M", "F"], size=20),
            "city": rng.choice(["Madrid", "Barcelona", "Valencia", "Sevilla"], size=20),
        }
    )


@pytest.fixture
def sample_config() -> dict:
    """Minimal valid METIS configuration dict."""
    return {
        "data": {
            "real": "data/real/test.csv",
            "synthetic": "data/synth/test.csv",
            "target": "score",
            "task_type": "regression",
            "schema": {
                "age": "continuous",
                "income": "continuous",
                "score": "continuous",
                "gender": "categorical",
                "city": "categorical",
            },
        },
        "evaluation": {
            "metric_ids": ["fidelity.ks", "fidelity.wasserstein"],
            "seed": 42,
            "cv_splits": 3,
            "n_runs": 1,
        },
        "reproducibility": {"seed": 42},
        "calibration": {"enabled": False},
        "aggregation": {"risk_aversion": 5.0},
        "report": {"output_dir": "reports/test", "formats": ["json"]},
    }


@pytest.fixture
def sample_eval_plan() -> EvalPlan:
    """EvalPlan with a small set of concrete metrics."""
    return EvalPlan(
        metric_ids=["fidelity.ks", "fidelity.wasserstein", "fidelity.hellinger"],
        seed=42,
        cv_splits=3,
    )


@pytest.fixture
def sample_metric_results() -> list[MetricResult]:
    """List of MetricResult objects spanning all families."""
    return [
        MetricResult(id="fidelity.ks", value=0.85, details={"p_value": 0.3}, family="fidelity"),
        MetricResult(id="fidelity.wasserstein", value=0.72, details={}, family="fidelity"),
        MetricResult(id="fidelity.hellinger", value=0.90, details={}, family="fidelity"),
        MetricResult(id="privacy.dcr", value=0.65, details={}, family="privacy"),
        MetricResult(id="privacy.nnaa", value=0.55, details={}, family="privacy"),
        MetricResult(id="utility.ml_efficiency", value=0.78, details={}, family="utility"),
    ]


@pytest.fixture
def sample_run_summary(sample_eval_plan, sample_metric_results) -> RunSummary:
    """Complete RunSummary for testing reporting/aggregation consumers."""
    return RunSummary(
        plan=sample_eval_plan,
        results=sample_metric_results,
        aggregates={
            "fidelity_score": 0.82,
            "privacy_score": 0.60,
            "utility_score": 0.78,
            "composite_score": 0.73,
            "total_metrics": 6,
            "successful_metrics": 6,
            "failed_metrics": 0,
        },
        artifacts={"seed": 42},
    )
