"""
TSTR - Train on Synthetic, Test on Real.

Strategy that evaluates practical utility of synthetic data
by training on synthetic and testing on real data.

Uses the **same** held-out real test split as TTR (same seed guarantees
the same ``train_test_split`` partition).  This ensures a fair
delta comparison: both strategies are evaluated on identical test rows.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseStrategyMetric, TrainingStrategy


class TSTRMetric(BaseStrategyMetric):
    """
    Train on Synthetic, Test on Real.

    The most important practical metric. Evaluates whether models
    trained on synthetic data can perform well on real-world data.
    This directly measures the utility of synthetic data as a
    training data substitute.

    Split: Synthetic (80%) → Train, Real (held-out 20%) → Test
    Train: 80 % of synthetic data (same split ratio/seed as TTR)
    Test: Same held-out real split used by TTR (same seed)

    Using the same split ratio and seed for both real and synthetic
    avoids data-leakage when ``synth == real`` (e.g. the ``real_data``
    baseline generator) and makes the training-volume comparison with
    TTR fair (80 % each).
    """

    strategy = TrainingStrategy.TSTR

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
        Prepare train from synthetic data, test from real hold-out.

        Both real and synthetic are split with the **same** ``test_size``
        and ``seed``.  Only the training portion (1 − test_size) of the
        synthetic data is used for training, and only the test portion of
        the real data is used for evaluation.

        This guarantees:
        * Fair comparison with TTR: same test set, same training volume.
        * No data leakage when ``synth == real``: the 80 % train and
          20 % test partitions are identical in that case, so the model
          never trains on the test rows.

        Args:
            real_data: Original dataset (held-out portion used for testing)
            synth_data: Synthetic dataset (train portion used for training)
            target_col: Target column name
            feature_cols: Feature column names
            test_size: Test set proportion (default 0.2, matches TTR)
            seed: Random seed (same as TTR for identical split)

        Returns:
            Dictionary with X_train from synth train split,
            X_test from real hold-out
        """
        # Split synthetic data — keep only the training portion
        X_train, _, y_train, _ = train_test_split(
            synth_data[feature_cols].copy(),
            synth_data[target_col].values,
            test_size=test_size,
            random_state=seed,
        )

        # Use the SAME held-out real test split as TTR (same seed → same partition)
        _, X_test, _, y_test = train_test_split(
            real_data[feature_cols].copy(),
            real_data[target_col].values,
            test_size=test_size,
            random_state=seed,
        )

        return {
            "X_train": X_train.reset_index(drop=True),
            "X_test": X_test.reset_index(drop=True),
            "y_train": y_train,
            "y_test": y_test,
        }

    def get_description(self) -> str:
        """Return human-readable description."""
        return "Train on Synthetic, Test on Real (Practical utility)"
        return "Train on Synthetic, Test on Real (Practical utility)"
