"""
TRTS - Train on Real, Test on Synthetic.

Strategy that evaluates distribution shift by training on real data
and testing on synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseStrategyMetric, TrainingStrategy, balance_datasets


class TRTSMetric(BaseStrategyMetric):
    """
    Train on Real, Test on Synthetic.

    Evaluates how well a model trained on real data performs
    on synthetic data. High performance indicates the synthetic
    data distribution is similar to the real data.

    Split: Real → Train, Synthetic → Test
    Train: Real train set
    Test: Synthetic test set
    """

    strategy = TrainingStrategy.TRTS

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
        Prepare train from real data, test from synthetic data.

        Args:
            real_data: Original dataset (for training)
            synth_data: Synthetic dataset (for testing)
            target_col: Target column name
            feature_cols: Feature column names
            test_size: Test set proportion (used for consistency)
            seed: Random seed

        Returns:
            Dictionary with X_train from real, X_test from synthetic
        """
        # Keep as DataFrame for CatBoost native categorical support
        X_real = real_data[feature_cols].copy()
        y_real = real_data[target_col].values

        X_synth = synth_data[feature_cols].copy()
        y_synth = synth_data[target_col].values

        # Balance datasets for fair comparison
        X_real, y_real, X_synth, y_synth = balance_datasets(X_real, y_real, X_synth, y_synth, seed)

        # Train set from real data
        X_train, _, y_train, _ = train_test_split(
            X_real, y_real, test_size=test_size, random_state=seed
        )

        # Test set from synthetic data
        _, X_test, _, y_test = train_test_split(
            X_synth, y_synth, test_size=test_size, random_state=seed
        )

        return {
            "X_train": X_train.reset_index(drop=True),
            "X_test": X_test.reset_index(drop=True),
            "y_train": y_train,
            "y_test": y_test,
        }

    def get_description(self) -> str:
        """Return human-readable description."""
        return "Train on Real, Test on Synthetic (Distribution shift)"
        return "Train on Real, Test on Synthetic (Distribution shift)"
