"""
TTS - Train on Synthetic, Test on Synthetic.

Strategy that evaluates synthetic data self-consistency
by training and testing entirely on synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseStrategyMetric, TrainingStrategy, subsample


class TTSMetric(BaseStrategyMetric):
    """
    Train on Synthetic, Test on Synthetic.

    Evaluates synthetic data internal consistency. High performance
    indicates the synthetic data has learnable patterns, but doesn't
    guarantee similarity to real data.

    Split: Synthetic → Train/Test
    Train: Synthetic train set
    Test: Synthetic test set
    """

    strategy = TrainingStrategy.TTS

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
        Prepare train/test splits from synthetic data only.

        Args:
            real_data: Original dataset (used to determine fair comparison limit)
            synth_data: Synthetic dataset
            target_col: Target column name(s)
            feature_cols: Feature column names
            test_size: Test set proportion
            seed: Random seed
            n_samples_limit: If provided, limit total samples to this number (Fair Comparison)

        Returns:
            Dictionary with X_train, X_test, y_train, y_test from synthetic data
        """
        # Keep as DataFrame for CatBoost native categorical support
        X = synth_data[feature_cols].copy()
        y = synth_data[target_col].values

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
        return "Train on Synthetic, Test on Synthetic (Self-consistency)"
        return "Train on Synthetic, Test on Synthetic (Self-consistency)"
