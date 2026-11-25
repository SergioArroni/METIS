"""Classification utility scored with the macro one-vs-rest ROC AUC (2026-10-19).

Weighted F1 thresholded at 0.5 hid the signal of imbalanced targets (rrhh,
16 % positives): almost every prediction fell under the cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.utility import ml_efficiency_metric as mem
from metis.infrastructure.metrics.utility.ml_efficiency.catboost_trainer import macro_ovr_auc
from metis.interface.sdk import Evaluator


def test_auc_counts_correctly_ordered_mixed_pairs():
    # Ana 0.72 and Luis 0.41 left, Marta, Pedro and Eva stayed: 6 of 6 pairs right.
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.72, 0.41, 0.35, 0.18, 0.09])
    proba = np.column_stack([1 - p, p])
    assert macro_ovr_auc(y, np.array([0, 1]), proba) == pytest.approx(1.0)
    p[1] = 0.30  # Luis under Marta: 5 of 6
    assert macro_ovr_auc(y, np.array([0, 1]), np.column_stack([1 - p, p])) == pytest.approx(5 / 6)


def test_a_class_the_model_never_saw_scores_chance():
    y = np.array([0, 1, 2, 2])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.5, 0.5]])
    # classes 0 and 1 perfectly ranked; class 2 unseen -> 0.5
    assert macro_ovr_auc(y, np.array([0, 1]), proba) == pytest.approx((1.0 + 1.0 + 0.5) / 3)


def test_no_signal_classification_is_one_half():
    assert mem._no_signal_score(np.array([0, 0, 1]), np.array([1, 0, 0]), "classification") == 0.5


def _imbalanced(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    risk = x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 1, n)
    return pd.DataFrame(
        {"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "y": np.where(risk > 1.4, "yes", "no")}
    )  # ~16 % "yes", like rrhh


def test_an_imbalanced_target_with_signal_is_measured():
    cfg = {
        "data": {
            "schema": dict.fromkeys("abc", "continuous") | {"y": "nominal"},
            "target": "y",
        },
        "metrics": ["utility.ml_efficiency"],
        "evaluation": {"n_runs": 1},
    }
    real, holdout = _imbalanced(500, 0), _imbalanced(500, 1)
    ev = Evaluator()
    signal = ev.measure_utility_signal(real, cfg, 0, holdout=holdout)
    assert signal["applicable"] is True
    (result,) = ev.evaluate(
        real, _imbalanced(500, 2), cfg, 0, holdout=holdout, utility_signal=signal
    ).results
    assert result.status == "ok" and result.value > 0.8
    assert result.details["metric"] == "ROC AUC (macro one-vs-rest)"
