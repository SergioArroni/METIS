"""CTGAN (Conditional Tabular GAN) synthetic data generator."""

import warnings

import numpy as np
import pandas as pd

from .base import BaseGenerator
from .gan_utils import DEFAULT_HIGH_CARDINALITY_THRESHOLD, filter_high_cardinality

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer

    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False


class CTGANGenerator(BaseGenerator):
    """
    CTGAN (Conditional Tabular GAN) generator.

    Uses conditional generative adversarial networks specifically designed
    for tabular data with mixed variable types.

    High-cardinality categorical columns (>100 unique values by default) are
    automatically excluded during training to prevent OOM, then filled from
    the real-data distribution after generation.

    Requires: sdv (Synthetic Data Vault)
    """

    def __init__(
        self,
        name: str = "CTGAN",
        epochs: int = 300,
        batch_size: int = 500,
        random_state: int | None = None,
        high_cardinality_threshold: int = DEFAULT_HIGH_CARDINALITY_THRESHOLD,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.high_cardinality_threshold = high_cardinality_threshold
        self._model = None
        self._dropped_cols: list[str] = []

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Fit CTGAN to real data."""
        if not SDV_AVAILABLE:
            raise ImportError("sdv is required for CTGANGenerator. Install with: pip install sdv")

        self._real_data = real_data.copy()
        self._all_columns = real_data.columns.tolist()
        self._categorical_columns = categorical_columns or []
        self._ordinal_columns = ordinal_columns or {}
        self._continuous_columns = continuous_columns or []

        # Filter high-cardinality categoricals to prevent OOM
        train_data, kept_cat_cols, self._dropped_cols = filter_high_cardinality(
            real_data.copy(),
            self._categorical_columns,
            threshold=self.high_cardinality_threshold,
        )
        ordinal_kept = {
            k: v for k, v in self._ordinal_columns.items() if k not in self._dropped_cols
        }

        # Create metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_data)

        for col in kept_cat_cols:
            if col in train_data.columns:
                metadata.update_column(col, sdtype="categorical")
        for col in ordinal_kept:
            if col in train_data.columns:
                metadata.update_column(col, sdtype="categorical")
        for col in self._continuous_columns:
            if col in train_data.columns:
                metadata.update_column(col, sdtype="numerical")

        self._model = CTGANSynthesizer(
            metadata=metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            cuda=False,
            verbose=False,
        )

        if self.random_state is not None:
            np.random.seed(self.random_state)
            try:
                import torch

                torch.manual_seed(self.random_state)
            except ImportError:
                pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(train_data)

        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using CTGAN."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        if self.random_state is not None:
            np.random.seed(self.random_state)
            try:
                import torch

                torch.manual_seed(self.random_state)
            except ImportError:
                pass

        synthetic = self._model.sample(num_rows=n_samples)

        # Fill dropped high-cardinality columns from real data distribution
        if self._dropped_cols:
            for col in self._dropped_cols:
                if col in self._real_data.columns:
                    synthetic[col] = (
                        self._real_data[col]
                        .sample(n=n_samples, replace=True, random_state=self.random_state)
                        .values
                    )

        final_cols = [c for c in self._all_columns if c in synthetic.columns]
        return synthetic[final_cols]
