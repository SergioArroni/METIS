"""
TTRS - Train on Real+Synthetic (cocktail), Test on Real.

Strategy that evaluates data augmentation utility by combining
real and synthetic data for training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseStrategyMetric, TrainingStrategy, balance_datasets


class TTRSMetric(BaseStrategyMetric):
    """
    Train on Real+Synthetic cocktail, Test on Real.

    Evaluates the value of synthetic data as a data augmentation
    technique. Combines real and synthetic data for training
    and tests on real data.

    Split: Real → Test, Real+Synthetic → Train
    Train: Real train set + Synthetic train set (concatenated)
    Test: Real test set
    """

    strategy = TrainingStrategy.TTRS

    def prepare_data(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        target_col: str | list[str],
        feature_cols: list[str],
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """
        Prepare train from real+synthetic cocktail, test from real.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            target_col: Target column name
            feature_cols: Feature column names
            test_size: Test set proportion
            seed: Random seed

        Returns:
            Dictionary with X_train from real+synthetic, X_test from real
        """
        # Keep as DataFrame for CatBoost native categorical support
        X_real = real_data[feature_cols].copy()
        y_real = real_data[target_col].values

        X_synth = synth_data[feature_cols].copy()
        y_synth = synth_data[target_col].values

        # Balance datasets for fair comparison
        X_real, y_real, X_synth, y_synth = balance_datasets(X_real, y_real, X_synth, y_synth, seed)

        X_train_real, X_test, y_train_real, y_test = train_test_split(
            X_real, y_real, test_size=test_size, random_state=seed
        )

        # Use all synthetic for training (already limited)
        X_train_synth, _, y_train_synth, _ = train_test_split(
            X_synth, y_synth, test_size=test_size, random_state=seed
        )

        # Combine real and synthetic for training (cocktail) - use pd.concat for DataFrames
        X_train = pd.concat([X_train_real, X_train_synth], ignore_index=True)
        y_train = np.concatenate([y_train_real, y_train_synth])

        # Shuffle the combined training data
        rng = np.random.default_rng(seed)
        shuffle_idx = rng.permutation(len(X_train))
        X_train = X_train.iloc[shuffle_idx].reset_index(drop=True)
        y_train = y_train[shuffle_idx]

        return {
            "X_train": X_train,
            "X_test": X_test.reset_index(drop=True),
            "y_train": y_train,
            "y_test": y_test,
        }

    def get_description(self) -> str:
        """Return human-readable description."""
        return "Train on Real+Synthetic cocktail, Test on Real (Augmentation)"
        return "Train on Real+Synthetic cocktail, Test on Real (Augmentation)"
