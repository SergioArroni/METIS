"""
CatBoost model trainer: CatBoost defaults with early stopping, no search.

Provides the common ML model logic used by both classification
and regression efficiency metrics.
Supports native categorical features in CatBoost.

Why no hyperparameter search
----------------------------
Utility is the *difference* TTR − TSTR between two models trained the same
way, not the absolute performance of one.  The gain that tuning brings over
CatBoost's defaults is modest on tabular data (McElfresh et al., 2023;
Erickson et al., 2025, TabArena) and largely cancels in that difference.
The Optuna search it replaces was not deterministic with parallel trials
(the same seed moved the score by up to 0.045), with 10 trials it was in
practice a random search over a hand-picked space, and reusing the tuning of
the first split on the others leaked a little of the TTR test rows.

Protocol per train/test split:

1. Fit CatBoost with its defaults (``MAX_ITERATIONS`` trees, learning rate
   chosen by CatBoost from the data size) on 80 % of the *training* rows and
   stop early on the other 20 %.  The test rows never take part.
2. Refit ``n_runs`` models on all the training rows with the number of trees
   that early stopping chose and the learning rate CatBoost picked in step 1
   (pinned: with fewer trees CatBoost would choose a larger one).
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from metis.infrastructure.metrics.base import _env_int

# Early stopping is scored on a validation slice carved out of the *training*
# data; the test set is only used for the final score (U2).
VALIDATION_FRACTION = 0.2
# Below this many training rows there is no room for a validation slice: the
# model is then fitted with CatBoost's defaults and no early stopping.
MIN_ROWS_FOR_VALIDATION = 10
# CatBoost's own default number of trees; early stopping cuts it.
MAX_ITERATIONS = 1000
# Patience, in trees.  Longer than CatBoost's ``od_wait`` default (20): the
# validation slice is small (20 % of a few hundred rows) and its loss noisy,
# so a short patience can stop on a fluctuation.  Costs at most 50 extra trees.
EARLY_STOPPING_ROUNDS = 50
# Categorical columns with up to 10 levels are one-hot encoded instead of
# target statistics (CTR), which were ~90 % of the cost of a fit on a few
# hundred rows.  The only departure from CatBoost's defaults.
ONE_HOT_MAX_SIZE = 10
# Floor of the probability given to the true class by the log-loss: a model
# that rules a class out (p = 0) would otherwise cost an infinite loss.
LOG_LOSS_EPS = 1e-6


class _ConstantClassifier:
    """Predicts the only class seen in training (CatBoost refuses one class)."""

    def __init__(self, label: Any):
        self._label = label
        self.classes_ = np.asarray([label])

    def predict(self, X) -> np.ndarray:
        return np.repeat(np.asarray([self._label]), len(X))

    def predict_proba(self, X) -> np.ndarray:
        return np.ones((len(X), 1))


def macro_ovr_auc(y_true: np.ndarray, classes: np.ndarray, proba: np.ndarray) -> float:
    """Macro one-vs-rest ROC AUC over the classes of *y_true*.

    *proba* holds one column per entry of *classes* (the model's classes).  A
    test class the model never saw scores a constant 0, i.e. AUC 0.5, as a
    model without signal does; a class that is every test row is skipped
    (no negatives to rank).  For a binary target both classes give the same
    AUC, so the macro mean is the usual binary AUC.
    """
    y_true = np.asarray(y_true)
    column = {str(c): j for j, c in enumerate(np.asarray(classes))}
    aucs = []
    for c in np.unique(y_true):
        positive = y_true == c
        if positive.all():
            continue
        j = column.get(str(c))
        score = proba[:, j] if j is not None else np.zeros(len(y_true))
        aucs.append(roc_auc_score(positive, score))
    return float(np.mean(aucs)) if aucs else 0.5


def multiclass_log_loss(y_true: np.ndarray, classes: np.ndarray, proba: np.ndarray) -> float:
    """Mean ``−log p(true class)``, with ``p`` floored at ``LOG_LOSS_EPS``.

    *proba* holds one column per entry of *classes* (the model's classes); a
    test class the model never saw gets probability 0, i.e. the floor.
    """
    y_true = np.asarray(y_true)
    column = {str(c): j for j, c in enumerate(np.asarray(classes))}
    idx = np.array([column.get(str(v), -1) for v in y_true])
    p = np.where(idx >= 0, proba[np.arange(len(y_true)), np.clip(idx, 0, None)], 0.0)
    return float(-np.mean(np.log(np.clip(p, LOG_LOSS_EPS, 1.0))))


def _fitted_learning_rate(model: Any) -> float | None:
    """Learning rate CatBoost chose (or was given) for a fitted *model*."""
    get_params = getattr(model, "get_all_params", None)
    if not callable(get_params):
        return None
    rate = get_params().get("learning_rate")
    return float(rate) if rate is not None else None


def _best_iteration(model: Any) -> int | None:
    get_best = getattr(model, "get_best_iteration", None)
    best = get_best() if callable(get_best) else None
    return int(best) if best is not None else None


class CatBoostTrainer(ABC):
    """
    Abstract CatBoost trainer: defaults with early stopping (module docstring).

    Subclasses implement classification or regression specific logic.
    """

    def __init__(self, n_runs: int = 3, thread_count: int | None = None):
        """
        Initialize trainer.

        Args:
            n_runs: Number of refits on the whole training set per split
            thread_count: CatBoost threads per model. ``None`` derives it from
                ``METIS_N_JOBS`` (see ``_threads``).
        """
        self.n_runs = n_runs
        self.thread_count = thread_count

    # ------------------------------------------------------------------
    # Thread budget
    # ------------------------------------------------------------------
    # ``METIS_N_JOBS`` is the total thread budget of the process and every
    # model gets all of it (models are fitted one after another; CatBoost
    # gives the same model with 1 or N threads).  Without it CatBoost keeps
    # its own default (all cores).  An explicit ``thread_count`` always wins.

    @staticmethod
    def _thread_budget() -> int:
        # A malformed METIS_N_JOBS used to raise inside every strategy (NaN utility).
        return max(1, _env_int("METIS_N_JOBS", 1))

    def _threads(self) -> int:
        if self.thread_count is not None:
            return self.thread_count
        return self._thread_budget() if "METIS_N_JOBS" in os.environ else -1

    def _runtime_params(self) -> dict[str, Any]:
        """Params that affect resources, never the fitted model."""
        # allow_writing_files=False: no ``catboost_info/`` in the cwd, which
        # parallel processes would otherwise race on.
        return {"thread_count": self._threads(), "allow_writing_files": False}

    @abstractmethod
    def _get_model_class(self):
        """Return the CatBoost model class."""
        pass

    @abstractmethod
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the evaluation metric."""
        pass

    def _secondary_metrics(self, y_true: np.ndarray, y_pred: Any) -> dict[str, float]:
        """Other scores of the same predictions (``last_secondary``); none by default."""
        return {}

    def _base_params(self, seed: int) -> dict[str, Any]:
        """CatBoost defaults plus the fixed choices documented above."""
        return {
            "iterations": MAX_ITERATIONS,
            "one_hot_max_size": ONE_HOT_MAX_SIZE,
            "random_seed": seed,
            "verbose": False,
        }

    def _get_categorical_indices(self, X: pd.DataFrame | np.ndarray) -> list[int]:
        """Get indices of categorical columns for CatBoost."""
        if isinstance(X, pd.DataFrame):
            cat_indices = []
            for i, col in enumerate(X.columns):
                if (
                    X[col].dtype == "object"
                    or X[col].dtype.name == "category"
                    or pd.api.types.is_string_dtype(X[col])
                ):
                    cat_indices.append(i)
            return cat_indices
        return []

    def _prepare_for_catboost(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
    ) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray, list[int]]:
        """
        Prepare data for CatBoost, handling categorical features.

        CatBoost handles categorical features natively, including normalization.
        We keep DataFrames as-is and let CatBoost handle them directly.

        Returns:
            tuple of (X_train_processed, X_test_processed, cat_feature_indices)
        """
        if isinstance(X_train, pd.DataFrame):
            # Get categorical column indices
            cat_indices = self._get_categorical_indices(X_train)

            # Categorical NaN is its own category ('missing'); numeric NaN is
            # left as is: CatBoost handles it natively (``nan_mode="Min"``),
            # where a fill with 0 made a gap look like a real value of 0.
            X_train_proc = X_train.copy()
            X_test_proc = X_test.copy()

            for col in X_train_proc.columns:
                if X_train_proc[col].dtype in [
                    "object",
                    "category",
                ] or pd.api.types.is_string_dtype(X_train_proc[col]):
                    # Use object dtype — CatBoost rejects StringArray
                    X_train_proc[col] = X_train_proc[col].fillna("missing").astype(object)
                    X_test_proc[col] = X_test_proc[col].fillna("missing").astype(object)

            return X_train_proc, X_test_proc, cat_indices
        # Fallback for numpy arrays - scale and return
        scaler = StandardScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test), []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    _stratify_validation = False

    def _fit(
        self,
        params: dict[str, Any],
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None = None,
    ):
        """Fit one model; with *eval_set*, stop early on it."""
        model = self._get_model_class()(**params)
        if eval_set is not None:
            try:
                model.fit(X, y, eval_set=eval_set, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                return model
            except Exception:  # e.g. a validation label unseen in training
                model = self._get_model_class()(**params)
        model.fit(X, y)
        return model

    def _validation_split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        seed: int,
    ) -> tuple[Any, Any, np.ndarray, np.ndarray] | None:
        """Split the training data into a fit part and a validation slice.

        Returns ``(X_fit, X_val, y_fit, y_val)``, or ``None`` when the table is
        too small for a slice.  Classification stratifies when every class
        has at least two rows.
        """
        if len(X) < MIN_ROWS_FOR_VALIDATION:
            return None
        stratify = None
        if self._stratify_validation and np.ndim(y) == 1:
            _, counts = np.unique(np.asarray(y).astype(str), return_counts=True)
            if len(counts) > 1 and counts.min() >= 2:
                stratify = y
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=VALIDATION_FRACTION, random_state=seed, stratify=stratify
        )
        if isinstance(X_fit, pd.DataFrame):
            X_fit, X_val = X_fit.reset_index(drop=True), X_val.reset_index(drop=True)
        return X_fit, X_val, y_fit, y_val

    def _settle(self, model: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Final params from a model early-stopped with *params*."""
        settled = dict(params)
        best = _best_iteration(model)
        if best is not None:
            settled["iterations"] = best + 1
        rate = _fitted_learning_rate(model)
        if rate is not None:
            settled["learning_rate"] = rate
        return settled

    def tune(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        seed: int = 42,
        cat_features: list[int] | None = None,
    ) -> dict[str, Any]:
        """Number of trees and learning rate, by early stopping on *X_train* only."""
        params = self._base_params(seed)
        X_proc, _, detected = self._prepare_for_catboost(X_train, X_train)
        cat_features = cat_features if cat_features is not None else detected
        split = self._validation_split(X_proc, y_train, seed)
        if split is None:
            return params
        X_fit, X_val, y_fit, y_val = split
        fit_params = {**params, **self._runtime_params()}
        if cat_features:
            fit_params["cat_features"] = cat_features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._fit(fit_params, X_fit, y_fit, (X_val, y_val))
        return self._settle(model, params)

    def _evaluate_with_params(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: dict[str, Any],
        cat_features: list[int] | None = None,
    ) -> list[float]:
        """Fit ``n_runs`` models on the whole training set and score them on the test set.

        The secondary scores of each run (``_secondary_metrics``) are left in
        ``last_secondary``: they come from the same fits, at no extra cost.
        """
        scores = []
        self.last_secondary: list[dict[str, float]] = []
        base_seed = params.get("random_seed", 42)

        X_train_proc, X_test_proc, detected_cat_features = self._prepare_for_catboost(
            X_train, X_test
        )
        cat_features = cat_features if cat_features is not None else detected_cat_features

        for i in range(self.n_runs):
            run_params = {**params, "random_seed": base_seed + i, **self._runtime_params()}
            if cat_features:
                run_params["cat_features"] = cat_features
            model = self._fit(run_params, X_train_proc, y_train)
            y_pred = self._predict(model, X_test_proc)
            scores.append(self._compute_metric(y_test, y_pred))
            self.last_secondary.append(self._secondary_metrics(y_test, y_pred))

        return scores

    def _predict(self, model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predictions that ``_compute_metric`` scores."""
        return model.predict(X)

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        seed: int = 42,
        cat_features: list[int] | None = None,
    ) -> tuple[dict[str, Any], list[float], float, float]:
        """
        Early-stop on a validation slice of the training data, then score on the test set.

        Args:
            X_train: Training features (DataFrame or ndarray)
            X_test: Test features (DataFrame or ndarray); only used for the
                final score, never to choose the number of trees
            y_train: Training labels
            y_test: Test labels
            seed: Random seed
            cat_features: Indices of categorical features (auto-detected if None)

        Returns:
            tuple of (final_params, scores_list, mean_score, std_score)
        """
        params = self.tune(X_train, y_train, seed, cat_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = self._evaluate_with_params(
                X_train, X_test, y_train, y_test, params, cat_features
            )
        return params, scores, float(np.mean(scores)), float(np.std(scores))


class ClassificationTrainer(CatBoostTrainer):
    """CatBoost classifier trainer scored with the macro one-vs-rest ROC AUC.

    The AUC ranks the predicted probabilities instead of thresholding them:
    with weighted F1 (until 2026-10-19) an imbalanced target such as rrhh
    (16 % positives) put almost every prediction under the 0.5 cut, the
    score barely beat predicting the majority class and its seed-to-seed
    noise hid the signal (signal-to-noise 0.35 against 1.9 with AUC; on
    telco 2.3 against 7.8, adult 3.8 against 14.3, cardio 7.4 against 7.2).
    """

    _stratify_validation = True

    def _get_model_class(self):
        return CatBoostClassifier

    def _fit(self, params, X, y, eval_set=None):
        if len(np.unique(np.asarray(y).astype(str))) < 2:
            return _ConstantClassifier(np.asarray(y)[0])
        return super()._fit(params, X, y, eval_set)

    def _predict(self, model: Any, X: pd.DataFrame | np.ndarray) -> Any:
        return np.asarray(model.classes_), np.asarray(model.predict_proba(X))

    def _compute_metric(self, y_true: np.ndarray, y_pred: Any) -> float:
        classes, proba = y_pred
        return macro_ovr_auc(y_true, classes, proba)

    def _secondary_metrics(self, y_true: np.ndarray, y_pred: Any) -> dict[str, float]:
        # The AUC ranks; the log-loss also scores how sure the model is
        # (``utility.log_loss``): label noise leaves the AUC and moves it.
        classes, proba = y_pred
        return {"log_loss": multiclass_log_loss(y_true, classes, proba)}


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa (Cohen, 1968) of two vectors of level ranks.

    With quadratic weights kappa equals Lin's concordance correlation of the
    ranks, ``2·cov / (var_t + var_p + (mean_t − mean_p)²)`` with population
    moments, which needs no level grid (only the ranks ``i/(K−1)``).  It is 0
    for a constant prediction or one independent of the truth, 1 for perfect
    agreement, and it grows with the distance between levels: one level off
    costs far less than the opposite end.
    """
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denom = t.var() + p.var() + (t.mean() - p.mean()) ** 2
    if denom <= 0:
        return 1.0 if np.allclose(t, p) else 0.0
    return float(2.0 * np.mean((t - t.mean()) * (p - p.mean())) / denom)


class OrdinalClassificationTrainer(ClassificationTrainer):
    """Classifier for an ordinal target, scored with QWK on the level ranks.

    The target is modelled as categories (a classifier beat a regression on
    the rank with ordinal yardsticks too), but the score must see the order:
    weighted F1 scored a one-level shift of the whole target 0.00 on cardio,
    below an independent target (0.42).  The prediction is the level closest
    to the expected rank under the predicted probabilities: the Bayes
    decision under the quadratic loss that QWK weighs (on Adult, K = 16, it
    gave QWK 0.52 against 0.43 with the most probable class).

    *level_ranks* maps each class code to its rank ``i/(K−1)``; a code
    without a rank (a synthetic label that is no real level) counts as the
    farthest level from the truth.
    """

    def __init__(self, level_ranks: dict[int, float], **kwargs: Any):
        super().__init__(**kwargs)
        self.level_ranks = {int(k): float(v) for k, v in level_ranks.items()}

    def _ranks(self, codes: np.ndarray) -> np.ndarray:
        return np.array([self.level_ranks.get(int(c), np.nan) for c in np.ravel(codes)])

    def _secondary_metrics(self, y_true: np.ndarray, y_pred: Any) -> dict[str, float]:
        # Its predictions are ranks, not probabilities: no log-loss.
        return {}

    def _predict(self, model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = getattr(model, "predict_proba", None)
        known = sorted(self.level_ranks.items(), key=lambda kv: kv[1])
        if not callable(proba) or not known:
            return np.ravel(model.predict(X))
        classes = np.asarray(model.classes_).astype(int)
        class_ranks = self._ranks(classes)
        usable = ~np.isnan(class_ranks)
        probs = np.asarray(proba(X))[:, usable]
        expected = (probs @ class_ranks[usable]) / np.maximum(probs.sum(axis=1), 1e-12)
        codes = np.array([c for c, _ in known])
        ranks = np.array([r for _, r in known])
        return codes[np.abs(expected[:, None] - ranks[None, :]).argmin(axis=1)]

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        t, p = self._ranks(y_true), self._ranks(y_pred)
        # A prediction that is no real level: as far as possible from the truth.
        p = np.where(np.isnan(p), np.where(t >= 0.5, 0.0, 1.0), p)
        keep = ~np.isnan(t)
        return quadratic_weighted_kappa(t[keep], p[keep])


class RegressionTrainer(CatBoostTrainer):
    """CatBoost regressor trained and scored with MAE.

    Trained with ``loss_function="MAE"`` (CatBoost's default is RMSE): the
    score is an MAE, and with a heavy-tailed target (Airbnb price) an
    RMSE-trained model barely beat predicting the median in MAE, so utility
    came out NaN ("no signal") or pure noise.  Early stopping follows the loss.
    """

    def _get_model_class(self):
        return CatBoostRegressor

    def _base_params(self, seed: int) -> dict[str, Any]:
        return {**super()._base_params(seed), "loss_function": "MAE"}

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)


# ── Multi-target trainers ────────────────────────────────────────────────


class MultiTargetRegressionTrainer(CatBoostTrainer):
    """CatBoost regressor with MultiRMSE for native multi-output regression."""

    def _get_model_class(self):
        return CatBoostRegressor

    def _base_params(self, seed: int) -> dict[str, Any]:
        return {**super()._base_params(seed), "loss_function": "MultiRMSE"}

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred, multioutput="uniform_average")


class MultiTargetClassificationTrainer(CatBoostTrainer):
    """One CatBoostClassifier per target for multi-target classification.

    Each sub-estimator handles categorical features natively via CatBoost
    and is early-stopped on its own: the settled params carry one
    ``(iterations, learning_rate)`` per target under ``"per_target"``.
    Scoring is the mean of the per-target macro one-vs-rest ROC AUC.
    """

    def _get_model_class(self):
        return CatBoostClassifier

    def _predict(self, model: Any, X: pd.DataFrame | np.ndarray) -> Any:
        return [(np.asarray(m.classes_), np.asarray(m.predict_proba(X))) for m in model.models]

    def _compute_metric(self, y_true: np.ndarray, y_pred: Any) -> float:
        # y_true is 2-D (n_samples, n_targets); y_pred one (classes, proba) per target
        y_true = np.asarray(y_true).reshape(len(y_true), -1)
        scores = [
            macro_ovr_auc(y_true[:, i], classes, proba) for i, (classes, proba) in enumerate(y_pred)
        ]
        return float(np.mean(scores))

    def _fit(self, params, X, y, eval_set=None):
        # One CatBoost per target, without ``MultiOutputClassifier``: sklearn's
        # ``clone`` rejects CatBoost when ``cat_features`` is set (the whole
        # strategy failed for any table with a categorical feature).
        params = dict(params)
        per_target = params.pop("per_target", None)
        models = []
        for j in range(y.shape[1]):
            y_j = y[:, j]
            if len(np.unique(np.asarray(y_j).astype(str))) < 2:
                models.append(_ConstantClassifier(y_j[0]))
                continue
            params_j = {**params, **(per_target[j] if per_target else {})}
            eval_j = (eval_set[0], eval_set[1][:, j]) if eval_set is not None else None
            models.append(super()._fit(params_j, X, y_j, eval_j))
        return _PerTargetModel(models)

    def _settle(self, model: Any, params: dict[str, Any]) -> dict[str, Any]:
        per_target = []
        for sub in model.models:
            settled = super()._settle(sub, {})
            per_target.append(settled)
        return {**params, "per_target": per_target}


class _PerTargetModel:
    """Stacks the predictions of one fitted classifier per target."""

    def __init__(self, models: list):
        self.models = models

    def predict(self, X) -> np.ndarray:
        return np.column_stack([np.ravel(m.predict(X)) for m in self.models])
