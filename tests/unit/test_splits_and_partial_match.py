"""The shared split of the real table and the partial-exact-match strategy of the attacks.

* ``split_real`` / ``split_parts`` (2026-11-11): the benchmark and the
  calibration split the real table with the same function, stratified by a
  classification target.
* ``partial_match`` / ``audit_strategies`` (2026-11-12): a generator that
  copies training rows and replaces part of their values (``delete_mean``)
  kept the three attacks at 1.0 with the whole-record distance alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.privacy.privacy_base import (
    audit_strategies,
    partial_match,
)
from metis.shared.splits import (
    CALIBRATION_SEED_OFFSET,
    config_split,
    split_parts,
    split_real,
    stratify_target,
)


def _table(n: int, positives: float = 0.2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "x": rng.normal(size=n),
            "y": (rng.random(n) < positives).astype(int),
        }
    )


class TestSplitReal:
    def test_parts_are_disjoint_and_cover_the_table(self):
        real = _table(1000)
        train, holdout, unseen = split_real(real, seed=42, stratify_by="y")
        ids = [set(p["id"]) for p in (train, holdout, unseen)]
        assert [len(p) for p in (train, holdout, unseen)] == [600, 200, 200]
        assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
        assert ids[0] | ids[1] | ids[2] == set(real["id"])

    def test_stratified_by_the_target(self):
        real = _table(1000, positives=0.16)
        rate = real["y"].mean()
        for part in split_real(real, seed=43, stratify_by="y"):
            assert abs(part["y"].mean() - rate) < 0.01

    def test_seed_decides_the_split(self):
        real = _table(500)
        a = split_real(real, seed=1, stratify_by="y")[0]
        b = split_real(real, seed=1, stratify_by="y")[0]
        c = split_real(real, seed=2, stratify_by="y")[0]
        assert a["id"].tolist() == b["id"].tolist()
        assert a["id"].tolist() != c["id"].tolist()

    def test_unknown_or_missing_target_is_not_stratified(self):
        real = _table(300)
        assert len(split_real(real, seed=0, stratify_by="nope")[0]) == 180
        assert len(split_real(real, seed=0, stratify_by=None)[0]) == 180

    @pytest.mark.parametrize("fractions", [(0.6, 0.2, 0.2), (1 / 3, 1 / 3, 1 / 3)])
    @pytest.mark.parametrize("rare", [3, 4, 5])
    def test_a_rare_class_does_not_break_the_split(self, fractions, rare):
        real = _table(300, positives=0.0)
        real.loc[: rare - 1, "y"] = 1
        for seed in range(20):
            parts = split_real(real, seed=seed, fractions=fractions, stratify_by="y")
            assert sum(len(p) for p in parts) == len(real)
            parts = split_parts(real, 50, seed, stratify_by="y", fractions=fractions)
            assert len(parts[2]) == 50

    def test_many_small_classes_do_not_break_the_split(self):
        real = pd.DataFrame({"id": np.arange(36), "y": np.repeat(np.arange(12), 3)})
        parts = split_real(real, seed=0, fractions=(0.8, 0.1, 0.1), stratify_by="y")
        assert sum(len(p) for p in parts) == 36

    def test_repeated_index_labels(self):
        real = pd.concat([_table(150, seed=1), _table(150, seed=2)])
        parts = split_real(real, seed=0, stratify_by="y")
        assert [len(p) for p in parts] == [180, 60, 60]
        assert len(split_parts(real, 40, 0, stratify_by="y")[0]) == 120

    def test_missing_target_values_are_a_class(self):
        real = _table(300).astype({"y": float})
        real.loc[:29, "y"] = np.nan
        parts = split_real(real, seed=0, stratify_by="y")
        assert [int(p["y"].isna().sum()) for p in parts] == [18, 6, 6]

    @pytest.mark.parametrize("n", [1000, 1001, 1002, 1004])
    @pytest.mark.parametrize("fractions", [(0.6, 0.2, 0.2), (1 / 3, 1 / 3, 1 / 3)])
    def test_whole_table_of_any_size(self, n, fractions):
        parts = split_parts(_table(n), n, seed=0, stratify_by="y", fractions=fractions)
        assert len(parts[2]) == int(n * fractions[2] + 1e-9)

    def test_bad_fractions(self):
        with pytest.raises(ValueError):
            split_real(_table(100), seed=0, fractions=(0.5, 0.5, 0.5))


class TestSplitParts:
    def test_default_proportions_give_3m_m_m(self):
        train, holdout, unseen = split_parts(_table(5000), 400, seed=0, stratify_by="y")
        assert [len(train), len(holdout), len(unseen)] == [1200, 400, 400]
        assert not set(train["id"]) & set(holdout["id"]) and not set(train["id"]) & set(unseen["id"])

    def test_thirds_give_equal_parts(self):
        parts = split_parts(_table(5000), 400, seed=0, stratify_by="y", fractions=(1 / 3, 1 / 3, 1 / 3))
        assert [len(p) for p in parts] == [400, 400, 400]

    def test_part_size_is_capped_by_the_table(self):
        parts = split_parts(_table(1000), 10_000, seed=0, stratify_by="y")
        assert [len(p) for p in parts] == [600, 200, 200]

    def test_too_few_rows(self):
        with pytest.raises(ValueError):
            split_parts(_table(3), 10, seed=0)


class TestConfigSplit:
    def test_classification_target_is_stratified(self):
        cfg = {"data": {"target": "y", "task_type": "classification"}}
        assert config_split(cfg) == ("y", (0.6, 0.2, 0.2))

    def test_regression_is_not(self):
        assert stratify_target({"target": "price", "task_type": "regression"}) is None

    def test_calibration_split_override(self):
        cfg = {"data": {"target": "y"}, "calibration": {"split": [0.5, 0.25, 0.25]}}
        assert config_split(cfg)[1] == (0.5, 0.25, 0.25)
        with pytest.raises(ValueError):
            config_split({"calibration": {"split": [0.5, 0.5]}})

    def test_calibration_seeds_are_apart_from_evaluation_seeds(self):
        assert CALIBRATION_SEED_OFFSET >= 1000


def _population(n: int, seed: int) -> pd.DataFrame:
    """Near-unique numbers and text next to common categories."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "income": np.round(rng.lognormal(10, 1, n), 2),
            "charge": np.round(rng.normal(60, 20, n), 2),
            "name": [f"p{v}" for v in rng.integers(0, 10**9, n)],
            "sex": rng.choice(["a", "b"], n),
            "plan": rng.choice(["x", "y", "z"], n),
        }
    )


COLUMNS = ["income", "charge", "name", "sex", "plan"]
NAMES = ["whole_record_closeness", "partial_exact_match"]


def _audit(train: pd.DataFrame, holdout: pd.DataFrame, synth: pd.DataFrame) -> dict:
    (bits_m, _), (bits_h, _) = partial_match([train, holdout], synth, pd.concat([train, holdout]), COLUMNS)
    flat = (np.zeros(len(train)), np.zeros(len(holdout)))  # a whole-record attack that sees nothing
    return audit_strategies([flat, (bits_m, bits_h)], NAMES)


class TestPartialMatch:
    def test_an_independent_sample_does_not_leak(self):
        for seed in range(5):
            train, holdout, synth = (_population(400, 10 * seed + k) for k in range(3))
            assert _audit(train, holdout, synth)["privacy_score"] == 1.0

    def test_noise_drawn_from_training_and_holdout_alike_does_not_leak(self):
        train, holdout = _population(400, 1), _population(400, 2)
        pool = pd.concat([train, holdout], ignore_index=True)
        rng = np.random.default_rng(0)
        synth = pd.DataFrame({c: rng.choice(pool[c].to_numpy(), 400) for c in COLUMNS})
        assert _audit(train, holdout, synth)["privacy_score"] == 1.0

    def test_training_rows_with_half_their_values_replaced_leak(self):
        train, holdout = _population(400, 1), _population(400, 2)
        synth = train.copy()
        rng = np.random.default_rng(0)
        for col in ("income", "charge"):
            hide = rng.random(len(synth)) < 0.5
            synth.loc[hide, col] = round(float(train[col].mean()), 2)
        synth.loc[rng.random(len(synth)) < 0.5, "name"] = "unknown"
        audit = _audit(train, holdout, synth)
        assert audit["strategy"] == "partial_exact_match"
        assert audit["privacy_score"] < 0.5

    def test_numbers_match_by_value_not_by_text(self):
        real = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["u", "v", "w", "x"]})
        synth = pd.DataFrame({"a": [1.0, 9.0], "b": ["zz", "zz"]})
        (bits, best), = partial_match([real], synth, real, ["a", "b"])
        assert bits[0] == pytest.approx(2.0) and best[0] == 0
        assert bits[1:].tolist() == [0.0, 0.0, 0.0]

    def test_missing_values_never_match(self):
        real = pd.DataFrame({"a": [np.nan, 1.0, 2.0], "b": [None, "u", "v"]})
        synth = pd.DataFrame({"a": [np.nan], "b": [None]})
        (bits, _), = partial_match([real], synth, real, ["a", "b"])
        assert bits.tolist() == [0.0, 0.0, 0.0]


class TestAuditStrategies:
    def test_keeps_the_strongest_strategy(self):
        rng = np.random.default_rng(0)
        none = (rng.normal(size=400), rng.normal(size=400))
        strong = (np.r_[np.full(80, 10.0), rng.normal(size=320)], rng.normal(size=400))
        audit = audit_strategies([none, strong], NAMES)
        assert audit["strategy"] == "partial_exact_match"
        assert audit["privacy_score"] < 1.0
        assert set(audit["strategies"]) == set(NAMES)

    def test_a_strategy_without_scores_does_not_hide_the_other(self):
        rng = np.random.default_rng(0)
        empty = (np.array([np.nan]), np.array([np.nan]))
        strong = (np.r_[np.full(80, 10.0), rng.normal(size=320)], rng.normal(size=400))
        audit = audit_strategies([empty, strong], NAMES)
        assert audit["strategy"] == "partial_exact_match"
        assert audit["privacy_score"] < 1.0


class TestKeysAcrossTables:
    """The same value has the same key whatever type each table holds it in."""

    def _bits(self, real: pd.DataFrame, synth: pd.DataFrame) -> list[float]:
        (bits, _), = partial_match([real], synth, real, list(real.columns))
        return bits.tolist()

    def test_number_against_its_text(self):
        real = pd.DataFrame({"a": [1, 2, 3, 4]})
        assert self._bits(real, pd.DataFrame({"a": ["1", " 2.0 ", "x", "9"]})) == [2.0, 2.0, 0.0, 0.0]
        assert self._bits(real.astype(str), pd.DataFrame({"a": [1.0, 2.0]})) == [2.0, 2.0, 0.0, 0.0]

    def test_boolean_against_zero_one(self):
        real = pd.DataFrame({"a": [True, False, True, False]})
        assert self._bits(real, pd.DataFrame({"a": [1]})) == [1.0, 0.0, 1.0, 0.0]

    def test_single_against_double_precision(self):
        real = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4]})
        synth = pd.DataFrame({"a": np.array([0.1, 0.3], dtype=np.float32)})
        assert self._bits(real, synth) == [2.0, 0.0, 2.0, 0.0]

    def test_members_and_non_members_with_different_types(self):
        train = pd.DataFrame({"a": pd.Categorical(["1", "2", "3", "4"])})
        holdout = pd.DataFrame({"a": [5, 6, 7, 8]})
        synth = pd.DataFrame({"a": [1, 2, 5, 6]})
        (bm, _), (bh, _) = partial_match([train, holdout], synth, pd.concat([train, holdout]), ["a"])
        assert bm.tolist() == [3.0, 3.0, 0.0, 0.0] and bh.tolist() == [3.0, 3.0, 0.0, 0.0]

    def test_text_that_looks_like_a_number_is_left_alone_when_every_table_is_text(self):
        real = pd.DataFrame({"zip": ["08001", "8001", "33001", "33002"]})
        assert self._bits(real, pd.DataFrame({"zip": ["8001"]})) == [0.0, 2.0, 0.0, 0.0]


class TestSyntheticCap:
    def test_best_row_is_an_index_of_the_whole_table(self):
        rng = np.random.default_rng(0)
        real = pd.DataFrame({"a": np.arange(50.0), "b": rng.integers(0, 3, 50)})
        synth = pd.concat([real] * 8, ignore_index=True)
        (bits, best), = partial_match([real], synth, real, ["a", "b"], max_synth=200)
        found = bits > 5
        assert found.any() and best.max() >= 200
        assert (synth["a"].to_numpy()[best[found]] == real["a"].to_numpy()[found]).all()

    def test_small_tables_are_searched_whole(self):
        real = pd.DataFrame({"a": np.arange(50.0)})
        (bits, best), = partial_match([real], real, real, ["a"])
        assert best.tolist() == list(range(50))


class TestOneSplitKey:
    def test_each_side_falls_back_to_the_other(self):
        from metis.shared.splits import split_fractions

        assert split_fractions({}, "calibration") == (0.6, 0.2, 0.2)
        only_bench = {"benchmark": {"split": [0.5, 0.25, 0.25]}}
        only_cal = {"calibration": {"split": [0.5, 0.25, 0.25]}}
        for cfg in (only_bench, only_cal):
            assert split_fractions(cfg, "calibration") == split_fractions(cfg, "benchmark") == (0.5, 0.25, 0.25)
        both = {"benchmark": {"split": [0.6, 0.2, 0.2]}, "calibration": {"split": [1 / 3, 1 / 3, 1 / 3]}}
        assert split_fractions(both, "benchmark") == (0.6, 0.2, 0.2)
        assert split_fractions(both, "calibration") == (1 / 3, 1 / 3, 1 / 3)
        with pytest.raises(ValueError):
            split_fractions({"calibration": {"split": "thirds"}})

    def test_the_cache_key_follows_the_split_in_use(self, tmp_path):
        import yaml

        from metis.calibrate.cache.fingerprint import compute_config_fingerprint

        def key(extra):
            path = tmp_path / "c.yaml"
            path.write_text(yaml.safe_dump({"metrics": ["fidelity.ks"], **extra}), encoding="utf-8")
            return compute_config_fingerprint(str(path))

        assert key({"benchmark": {"split": [0.5, 0.25, 0.25]}}) == key({"calibration": {"split": [0.5, 0.25, 0.25]}})
        assert key({"benchmark": {"split": [0.5, 0.25, 0.25]}}) != key({})
        assert key({"benchmark": {"n_runs": 5}}) == key({})

    def test_regression_is_not_stratified_however_it_is_written(self):
        assert stratify_target({"target": "price", "task_type": "Regression"}) is None
