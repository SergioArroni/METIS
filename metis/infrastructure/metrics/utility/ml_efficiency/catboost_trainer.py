"""
CatBoost model trainer with Optuna optimization.

Provides the common ML model logic used by both classification
and regression efficiency metrics.
Supports native categorical features in CatBoost.
"""

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

# Silence Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class CatBoostTrainer(ABC):
    """
    Abstract CatBoost trainer with Optuna hyperparameter optimization.

    Based on Gorishniy et al., 2021 configuration.
    Subclasses implement classification or regression specific logic.
    """

    def __init__(self, n_trials: int = 100, n_runs: int = 3):
        """
        Initialize trainer.

        Args:
            n_trials: Number of Optuna trials for hyperparameter search
            n_runs: Number of evaluation runs with best parameters
        """
        self.n_trials = n_trials
        self.n_runs = n_runs

    @abstractmethod
    def _get_model_class(self):
        """Return the CatBoost model class."""
        pass

    @abstractmethod
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the evaluation metric."""
        pass

    @abstractmethod
    def _optimization_direction(self) -> str:
        """Return 'maximize' or 'minimize' for Optuna."""
        pass

    def _get_hyperparameter_space(self, trial: optuna.Trial, seed: int) -> dict[str, Any]:
        """
        Get hyperparameter search space based on Gorishniy et al., 2021.

        Hyperparameters configuration:
        - depth (max_depth): Uniform[3, 10] - Controls tree depth for complex interactions
        - learning_rate: LogUniform[1e-5, 1] - Step size for gradient updates
        - bagging_temperature: Uniform[0, 1] - Controls randomness in bagging
        - l2_leaf_reg: LogUniform[1, 10] - L2 regularization on leaf weights
        - leaf_estimation_iterations: Uniform[1, 10] - Number of gradient iterations per leaf
        - n_estimators fixed at 100 for consistent tuning trials
        """
        return {
            "depth": trial.suggest_int("depth", 3, 10),  # Uniform[3, 10]
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 1, log=True
            ),  # LogUniform[1e-5, 1]
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0, 1
            ),  # Uniform[0, 1]
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),  # LogUniform[1, 10]
            "leaf_estimation_iterations": trial.suggest_int(
                "leaf_estimation_iterations", 1, 10
            ),  # Uniform[1, 10]
            "iterations": 100,  # Number of tuning trials as per Gorishniy et al., 2021
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

            # Fill NaN for numeric columns with 0, categorical with 'missing'
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
                else:
                    X_train_proc[col] = X_train_proc[col].fillna(0)
                    X_test_proc[col] = X_test_proc[col].fillna(0)

            return X_train_proc, X_test_proc, cat_indices
        # Fallback for numpy arrays - scale and return
        scaler = StandardScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test), []

    def _create_objective(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        cat_features: list[int] = None,
    ) -> Callable:
        """Create Optuna objective function."""
        # Prepare data once outside the objective
        X_train_proc, X_test_proc, detected_cat_features = self._prepare_for_catboost(
            X_train, X_test
        )
        cat_features = cat_features if cat_features is not None else detected_cat_features

        def objective(trial: optuna.Trial) -> float:
            params = self._get_hyperparameter_space(trial, seed)
            if cat_features:
                params["cat_features"] = cat_features

            model = self._get_model_class()(**params)
            model.fit(X_train_proc, y_train)
            predictions = model.predict(X_test_proc)

            metric = self._compute_metric(y_test, predictions)

            # For minimization metrics, return negative
            if self._optimization_direction() == "minimize":
                return -metric
            return metric

        return objective

    def _evaluate_with_params(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: dict[str, Any],
        cat_features: list[int] = None,
    ) -> list[float]:
        """Evaluate model multiple times with given parameters."""
        scores = []
        base_seed = params.get("random_seed", 42)

        # Prepare data once
        X_train_proc, X_test_proc, detected_cat_features = self._prepare_for_catboost(
            X_train, X_test
        )
        cat_features = cat_features if cat_features is not None else detected_cat_features

        for i in range(self.n_runs):
            # Change seed for each run to get different results
            run_params = params.copy()
            run_params["random_seed"] = base_seed + i
            if cat_features:
                run_params["cat_features"] = cat_features

            model = self._get_model_class()(**run_params)
            model.fit(X_train_proc, y_train)
            predictions = model.predict(X_test_proc)

            score = self._compute_metric(y_test, predictions)
            scores.append(score)

        return scores

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        seed: int = 42,
        cat_features: list[int] = None,
    ) -> tuple[dict[str, Any], list[float], float, float]:
        """
        Run full optimization and evaluation pipeline.

        Args:
            X_train: Training features (DataFrame or ndarray)
            X_test: Test features (DataFrame or ndarray)
            y_train: Training labels
            y_test: Test labels
            seed: Random seed
            cat_features: Indices of categorical features (auto-detected if None)

        Returns:
            tuple of (best_params, scores_list, mean_score, std_score)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Detect categorical features if X is DataFrame
            detected_cat_features = self._get_categorical_indices(X_train)
            cat_features = cat_features if cat_features is not None else detected_cat_features

            # Optimize hyperparameters
            n_jobs = int(os.environ.get("METIS_N_JOBS", "1"))
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            objective = self._create_objective(X_train, X_test, y_train, y_test, seed, cat_features)
            study.optimize(
                objective,
                n_trials=self.n_trials,
                n_jobs=n_jobs,
                show_progress_bar=False,
            )

            best_params = study.best_params
            best_params["random_seed"] = seed
            best_params["verbose"] = False

            # Evaluate with best params
            scores = self._evaluate_with_params(
                X_train, X_test, y_train, y_test, best_params, cat_features
            )

            return best_params, scores, float(np.mean(scores)), float(np.std(scores))


class ClassificationTrainer(CatBoostTrainer):
    """CatBoost classifier trainer using F1 score."""

    def _get_model_class(self):
        return CatBoostClassifier

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average="weighted")

    def _optimization_direction(self) -> str:
        return "maximize"


class RegressionTrainer(CatBoostTrainer):
    """CatBoost regressor trainer using MAE."""

    def _get_model_class(self):
        return CatBoostRegressor

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)

    def _optimization_direction(self) -> str:
        return "minimize"


# ── Multi-target trainers ────────────────────────────────────────────────


class MultiTargetRegressionTrainer(CatBoostTrainer):
    """CatBoost regressor with MultiRMSE for native multi-output regression."""

    def _get_model_class(self):
        return CatBoostRegressor

    def _get_hyperparameter_space(self, trial: optuna.Trial, seed: int) -> dict[str, Any]:
        params = super()._get_hyperparameter_space(trial, seed)
        params["loss_function"] = "MultiRMSE"
        return params

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred, multioutput="uniform_average")

    def _optimization_direction(self) -> str:
        return "minimize"


class MultiTargetClassificationTrainer(CatBoostTrainer):
    """MultiOutputClassifier wrapping CatBoostClassifier for multi-target classification.

    Each sub-estimator handles categorical features natively via CatBoost.
    Scoring is the mean of per-target weighted F1.
    """

    def _get_model_class(self):
        return CatBoostClassifier

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # y_true / y_pred are 2-D (n_samples, n_targets)
        if y_true.ndim == 1:
            return f1_score(y_true, y_pred, average="weighted")
        scores = [
            f1_score(y_true[:, i], y_pred[:, i], average="weighted") for i in range(y_true.shape[1])
        ]
        return float(np.mean(scores))

    def _optimization_direction(self) -> str:
        return "maximize"

    # Override the full pipeline to wrap CatBoost inside MultiOutputClassifier
    def _create_objective(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        cat_features: list[int] = None,
    ) -> Callable:
        X_train_proc, X_test_proc, detected_cat_features = self._prepare_for_catboost(
            X_train, X_test
        )
        cat_features = cat_features if cat_features is not None else detected_cat_features

        def objective(trial: optuna.Trial) -> float:
            params = self._get_hyperparameter_space(trial, seed)
            if cat_features:
                params["cat_features"] = cat_features

            base_model = self._get_model_class()(**params)
            model = MultiOutputClassifier(base_model)
            model.fit(X_train_proc, y_train)
            predictions = model.predict(X_test_proc)

            metric = self._compute_metric(y_test, predictions)
            if self._optimization_direction() == "minimize":
                return -metric
            return metric

        return objective

    def _evaluate_with_params(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: dict[str, Any],
        cat_features: list[int] = None,
    ) -> list[float]:
        scores = []
        base_seed = params.get("random_seed", 42)

        X_train_proc, X_test_proc, detected_cat_features = self._prepare_for_catboost(
            X_train, X_test
        )
        cat_features = cat_features if cat_features is not None else detected_cat_features

        for i in range(self.n_runs):
            run_params = params.copy()
            run_params["random_seed"] = base_seed + i
            if cat_features:
                run_params["cat_features"] = cat_features

            base_model = self._get_model_class()(**run_params)
            model = MultiOutputClassifier(base_model)
            model.fit(X_train_proc, y_train)
            predictions = model.predict(X_test_proc)

            score = self._compute_metric(y_test, predictions)
            scores.append(score)

        return scores
