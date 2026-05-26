"""
TTR - Train on Real, Test on Real.

Baseline strategy that establishes reference performance
by training and testing on the original real dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseStrategyMetric, TrainingStrategy, subsample


class TTRMetric(BaseStrategyMetric):
    """
    Train on Real, Test on Real (Baseline).

    This is the reference strategy against which all other strategies
    are compared. It represents the best achievable performance when
    using only real data.

    Split: Real → Train/Test
    Train: Real train set
    Test: Real test set
    """

    strategy = TrainingStrategy.TTR

    def prepare_data(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        target_col: str | list[str],
        feature_cols: list[str],
        test_size: float = 0.2,
        seed: int = 42,
        n_samples_limit: int = None,
    ) -> dict[str, np.ndarray]:
        """
        Prepare train/test splits from real data only.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset (not used for TTR)
            target_col: Target column name(s)
            feature_cols: Feature column names
            test_size: Test set proportion
            seed: Random seed
            n_samples_limit: If provided, limit total samples to this number (Fair Comparison)

        Returns:
            Dictionary with X_train, X_test, y_train, y_test from real data
        """
        # Keep as DataFrame for CatBoost native categorical support
        X = real_data[feature_cols].copy()
        y = real_data[target_col].values

        if n_samples_limit:
            X, y = subsample(X, y, n_samples_limit, seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        return {
            "X_train": X_train.reset_index(drop=True),
            "X_test": X_test.reset_index(drop=True),
            "y_train": y_train,
            "y_test": y_test,
        }

    def get_description(self) -> str:
        """Return human-readable description."""
        return "Train on Real, Test on Real (Baseline)"
        return "Train on Real, Test on Real (Baseline)"
