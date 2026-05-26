"""ADSGAN (Anonymization through Data Synthesis GAN) synthetic data generator."""

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


class ADSGANGenerator(BaseGenerator):
    """
    ADSGAN (Anonymization through Data Synthesis GAN) generator.

    GAN-based approach with additional privacy-preserving mechanisms.

    Note: This is a placeholder implementation. The actual ADSGAN requires
    a custom implementation based on the original paper or external repository.
    """

    def __init__(
        self,
        name: str = "ADSGAN",
        epochs: int = 300,
        batch_size: int = 500,
        epsilon: float = 1.0,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.random_state = random_state
        self._model = None
        self._dropped_cols: list[str] = []

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Fit ADSGAN to real data."""
        warnings.warn(
            "ADSGAN is not fully implemented. Using CTGAN as fallback. "
            "To implement ADSGAN, add the custom implementation here or "
            "integrate an external library.",
            stacklevel=2,
        )

        if not SDV_AVAILABLE:
            raise ImportError("sdv is required for ADSGAN fallback. Install with: pip install sdv")

        self._real_data = real_data.copy()
        self._all_columns = real_data.columns.tolist()
        self._categorical_columns = categorical_columns or []
        self._ordinal_columns = ordinal_columns or {}
        self._continuous_columns = continuous_columns or []

        # Filter high-cardinality categoricals to prevent OOM
        train_data, kept_cat_cols, self._dropped_cols = filter_high_cardinality(
            real_data.copy(),
            self._categorical_columns,
            threshold=DEFAULT_HIGH_CARDINALITY_THRESHOLD,
        )

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_data)

        for col in kept_cat_cols:
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
        """Generate synthetic data using ADSGAN."""
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
        return synthetic[final_cols]
