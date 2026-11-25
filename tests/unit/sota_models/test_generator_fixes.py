"""Benchmark generators that did not do what the paper says (review 2026-10-09)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.interface.sdk import Evaluator
from metis.sota_models.comparison.orchestrator import split_real
from metis.sota_models.generators.cart import CARTGenerator
from metis.sota_models.generators.delete_impute import DeleteImputeZeroGenerator
from metis.sota_models.generators.real_data import RealDataGenerator
from metis.sota_models.generators.smotenc import SMOTENCGenerator


def _table(n: int = 400, seed: int = 0, p_yes: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.gamma(2.0, 2.0, size=n)
    c = rng.choice(["a", "b", "c"], size=n)
    lvl = rng.choice(["low", "mid", "high"], size=n)
    y = np.where(rng.random(n) < p_yes, "yes", "no")
    return pd.DataFrame({"x0": x0, "x1": x1, "c": c, "lvl": lvl, "y": y})


_ORD = {"lvl": ["low", "mid", "high"]}


def _real_rows(synth: pd.DataFrame, real: pd.DataFrame) -> float:
    keys = set(map(tuple, real.astype(str).to_numpy()))
    return float(np.mean([tuple(r) in keys for r in synth.astype(str).to_numpy()]))


# ── B1 SMOTENC ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("p_yes", [0.5, 0.13])
def test_smotenc_synthesises_new_rows_and_keeps_the_target_marginal(p_yes):
    real = _table(p_yes=p_yes)
    gen = SMOTENCGenerator(target_column="y", random_state=0)
    gen.fit(
        real, categorical_columns=["c", "y"], ordinal_columns=_ORD, continuous_columns=["x0", "x1"]
    )
    synth = gen.generate(400)
    assert len(synth) == 400
    assert _real_rows(synth, real) == 0.0  # used to be 94 % / 58 % real rows
    real_rate = (real["y"] == "yes").mean()
    assert abs((synth["y"] == "yes").mean() - real_rate) < 0.06  # not forced to 50/50
    assert set(synth["lvl"]) <= set(_ORD["lvl"])


def test_smotenc_regression_does_not_copy_feature_rows():
    real = _table()
    real["price"] = np.exp(real["x0"]) * 50
    gen = SMOTENCGenerator(target_column="price", task_type="regression", random_state=0)
    gen.fit(
        real.drop(columns="y"),
        categorical_columns=["c"],
        ordinal_columns=_ORD,
        continuous_columns=["x0", "x1", "price"],
    )
    synth = gen.generate(300)
    features = ["x0", "x1", "c", "lvl"]
    assert _real_rows(synth[features], real[features]) == 0.0


# ── B2 CART ───────────────────────────────────────────────────────────────


def test_cart_samples_the_leaves_instead_of_collapsing_to_their_mean():
    real = _table(1000)
    gen = CARTGenerator(random_state=0)
    gen.fit(
        real, categorical_columns=["c", "y"], ordinal_columns=_ORD, continuous_columns=["x0", "x1"]
    )
    synth = gen.generate(1000)
    for col in ("x0", "x1"):
        assert synth[col].nunique() > 300  # used to be 3 distinct values
        assert 0.7 < synth[col].std() / real[col].std() < 1.3  # used to be 0
    assert set(synth["c"]) == set(real["c"])


# ── B3 delete_zero ────────────────────────────────────────────────────────


def test_delete_zero_imputes_valid_values_and_metis_keeps_every_row():
    real = _table()
    gen = DeleteImputeZeroGenerator(deletion_rate=0.5, random_state=0)
    gen.fit(
        real, categorical_columns=["c", "y"], ordinal_columns=_ORD, continuous_columns=["x0", "x1"]
    )
    synth = gen.generate(400)
    assert set(synth["lvl"]) <= set(_ORD["lvl"])
    cfg = {
        "data": {
            "schema": {
                "x0": "continuous",
                "x1": "continuous",
                "c": "nominal",
                "lvl": {"type": "ordinal", "levels": _ORD["lvl"]},
                "y": "nominal",
            }
        },
        "metrics": ["fidelity.marginal.coverage"],
        "evaluation": {"n_runs": 1},
        "reproducibility": {"seed": 0},
    }
    summary = Evaluator().evaluate(real, synth, cfg)
    assert summary.artifacts["synth_rows_dropped"] == 0  # 88.8 % before


# ── real_data ─────────────────────────────────────────────────────────────


def test_real_data_returns_unseen_rows_and_refuses_to_copy_the_training_part():
    real = _table()
    gen = RealDataGenerator(random_state=0)
    with pytest.raises(ValueError, match="set_unseen"):
        gen.fit(real)
    unseen = _table(seed=9)
    gen.set_unseen(unseen)
    gen.fit(real)
    out = gen.generate(200)
    assert _real_rows(out, unseen) == 1.0
    assert _real_rows(out, real) == 0.0


def test_benchmark_split_is_disjoint_stratified_and_seeded():
    real = _table(600, p_yes=0.2)
    real["row"] = range(len(real))
    a = split_real(real, 1, stratify_by="y")
    b = split_real(real, 1, stratify_by="y")
    rows = [set(p["row"]) for p in a]
    assert sum(len(r) for r in rows) == 600 and not (rows[0] & rows[1] or rows[0] & rows[2])
    assert all(abs((p["y"] == "yes").mean() - 0.2) < 0.03 for p in a)
    assert all(x.equals(y) for x, y in zip(a, b, strict=True))


# ── B5 BN ─────────────────────────────────────────────────────────────────


def test_bn_turns_bins_into_observed_values_without_extrapolating_the_tail():
    pytest.importorskip("pgmpy")
    from metis.sota_models.generators.bayesian_network import (  # noqa: PLC0415
        BayesianNetworkGenerator,
    )

    real = _table(800)
    real["x1"] = np.exp(real["x0"] * 2)  # heavy tail
    gen = BayesianNetworkGenerator(random_state=0)
    gen.fit(
        real, categorical_columns=["c", "y"], ordinal_columns=_ORD, continuous_columns=["x0", "x1"]
    )
    synth = gen.generate(800)
    assert set(synth["x1"]) <= set(real["x1"])  # every value is an observed one
    # The top quintile used to be drawn uniformly up to the maximum.
    assert synth["x1"].mean() < 2 * real["x1"].mean()
    assert list(synth.columns) == list(real.columns)


# ── B4 DP-CTGAN ───────────────────────────────────────────────────────────


def test_dpctgan_spends_only_its_budget_and_never_copies_real_values():
    pytest.importorskip("snsynth")
    from metis.sota_models.generators.dpctgan import SUPPRESSED, DPCTGANGenerator  # noqa: PLC0415

    real = _table(1000)
    real["host_name"] = [f"host_{i}" for i in range(1000)]  # high cardinality, private labels
    real.loc[3, "x1"] = np.nan  # a NaN used to break the discriminator
    gen = DPCTGANGenerator(
        epsilon=3.0,
        epochs=1,
        batch_size=500,
        random_state=0,
        high_cardinality_threshold=20,
        bounds={"x0": (-5.0, 5.0)},
    )
    gen.fit(
        real,
        categorical_columns=["c", "y", "host_name"],
        ordinal_columns=_ORD,
        continuous_columns=["x0", "x1"],
    )
    acc = gen.privacy_accounting
    assert acc["batch_size"] <= acc["rows_used"] and acc["batch_size"] % 2 == 0
    assert acc["public_bounds"] == ["x0"] and acc["inferred_bounds"] == ["x1"]
    spent = acc["epsilon_histograms"] + acc["epsilon_bounds_inference"] + acc["epsilon_dp_sgd"]
    assert spent == pytest.approx(3.0)
    assert acc["suppressed_columns"] == ["host_name"]
    synth = gen.generate(50)
    assert set(synth["host_name"]) == {SUPPRESSED}  # real host names used to be copied
