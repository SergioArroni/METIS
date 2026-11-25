"""fidelity.discriminator and utility.log_loss (2026-11-03)."""

import numpy as np
import pandas as pd
import pytest

from metis.domain.taxonomy import FIDELITY_METRICS, UTILITY_METRICS
from metis.infrastructure.metrics.fidelity.global_metrics.detection.discriminator import DiscriminatorMetric
from metis.infrastructure.metrics.utility.ml_efficiency.base import (
    StrategyResult,
    TrainingStrategy,
)
from metis.infrastructure.metrics.utility.ml_efficiency.catboost_trainer import (
    LOG_LOSS_EPS,
    multiclass_log_loss,
)
from metis.infrastructure.metrics.utility.ml_efficiency_metric import (
    _gains,
    _no_signal_score,
    _view,
    pooled_utility_signal,
)


def _table(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1.0, n)
    return pd.DataFrame(
        {
            "x": x,
            "y": x + rng.normal(0, 0.5, n),
            "c": np.where(x + rng.normal(0, 0.5, n) > shift, "a", "b"),
        }
    )


def test_new_metrics_have_their_own_categories() -> None:
    assert FIDELITY_METRICS["global"]["subcategories"]["detection"] == ["fidelity.discriminator"]
    assert UTILITY_METRICS["probabilistic"]["metrics"] == ["utility.log_loss"]


def test_discriminator_cannot_tell_two_samples_of_one_table() -> None:
    result = DiscriminatorMetric().fit(_table(600, 0), _table(600, 1), {"seed": 0}).compute()
    assert result.details["auc"] < 0.6
    assert result.value > 0.8


def test_discriminator_sees_a_broken_joint_with_the_marginals_kept() -> None:
    real = _table(600, 0)
    synth = _table(600, 1)
    # Each column shuffled on its own: same marginals, no dependence left.
    rng = np.random.default_rng(2)
    for c in synth.columns:
        synth[c] = rng.permutation(synth[c].to_numpy())
    result = DiscriminatorMetric().fit(real, synth, {"seed": 0}).compute()
    assert result.details["auc"] > 0.75
    assert result.value == pytest.approx(np.clip(2 * (1 - result.details["auc"]), 0, 1))
    assert set(result.details["top_columns"]) <= {"x", "y", "c"}


def test_log_loss_floors_the_true_class_probability() -> None:
    classes = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1, 1])
    expected = -np.mean(np.log([0.9, 1.0, LOG_LOSS_EPS]))
    assert multiclass_log_loss(y, classes, proba) == pytest.approx(expected)
    # A class the model never saw gets the floor too.
    assert multiclass_log_loss(np.array([2]), classes, proba[:1]) == pytest.approx(
        -np.log(LOG_LOSS_EPS)
    )


def test_log_loss_without_signal_is_the_entropy_of_the_training_frequencies() -> None:
    y_train = np.array([0, 0, 0, 1])
    y_test = np.array([0, 1])
    assert _no_signal_score(y_train, y_test, "log_loss") == pytest.approx(
        -np.mean(np.log([0.75, 0.25]))
    )
    # Lower is better: the gain over the model without signal is m0 − m.
    assert _gains([0.3], [0.5], "log_loss")[0] == pytest.approx(0.2)


def test_view_scores_the_same_fits_with_the_secondary_score() -> None:
    fits = StrategyResult(
        strategy=TrainingStrategy.TSTR,
        score=0.8,
        raw_metric=0.8,
        scores_per_run=[0.8, 0.8],
        std=0.0,
        best_params={"iterations": 10},
        extra_scores_per_run={"log_loss": [0.4, 0.6]},
    )
    view = _view(fits, "log_loss")
    assert view.scores_per_run == [0.4, 0.6]
    assert view.score == pytest.approx(0.5)
    assert view.best_params == {"iterations": 10}
    assert not _view(fits, "brier").is_valid


def _probe(task_type, chance, ttr, half, rows):
    return {
        "task_type": task_type,
        "no_signal_scores_per_split": chance,
        "ttr_scores_per_split": ttr,
        "ttr_half_sample_scores_per_split": half,
        "rows": rows,
    }


def test_pooled_signal_carries_the_log_loss_scale() -> None:
    rows = {"real": 100, "holdout": 100}
    probe = _probe("classification", [0.5] * 4, [0.80, 0.82, 0.79, 0.81], [0.78, 0.80, 0.78, 0.80], rows)
    probe["log_loss"] = _probe(
        "log_loss", [0.69] * 4, [0.45, 0.44, 0.46, 0.45], [0.47, 0.46, 0.47, 0.48], rows
    )
    signal = pooled_utility_signal([probe])
    assert signal["applicable"]
    log_loss = signal["log_loss"]
    assert log_loss["task_type"] == "log_loss"
    assert log_loss["ttr_gain"] == pytest.approx(0.69 - 0.45)
    assert log_loss["rows"] == rows


def test_log_loss_does_not_apply_where_the_auc_does_not() -> None:
    rows = {"real": 100, "holdout": 100}
    # The AUC barely beats chance: no signal for the task.
    probe = _probe("classification", [0.5] * 4, [0.51, 0.49, 0.52, 0.50], [0.50, 0.52, 0.49, 0.51], rows)
    probe["log_loss"] = _probe(
        "log_loss", [0.69] * 4, [0.45, 0.44, 0.46, 0.45], [0.47, 0.46, 0.47, 0.48], rows
    )
    signal = pooled_utility_signal([probe])
    assert not signal["applicable"]
    assert not signal["log_loss"]["applicable"]
    assert signal["log_loss"]["reason"] == signal["reason"]
