"""TVAE (Tabular Variational Autoencoder) synthetic data generator."""

import warnings

import numpy as np
import pandas as pd

from .base import BaseGenerator
from .gan_utils import DEFAULT_HIGH_CARDINALITY_THRESHOLD, collapse_high_cardinality

try:
    from sdv.single_table import TVAESynthesizer

    from . import _sdv

    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False


class TVAEGenerator(BaseGenerator):
    """
    TVAE (Tabular Variational Autoencoder) generator.

    Uses variational autoencoders specifically designed for tabular data
    with mixed variable types, as described in *Modeling Tabular Data
    using Conditional GAN* (Xu et al., NeurIPS 2019).

    High-cardinality categorical columns (>100 unique values by default) keep
    their 100 most frequent categories and map the rest to ``OTHER`` before
    training (one-hot stays small), and are generated jointly with the rest.
    They are never refilled with real values.

    Requires: sdv (Synthetic Data Vault)
    """

    def __init__(
        self,
        name: str = "TVAE",
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
        """Fit TVAE to real data."""
        if not SDV_AVAILABLE:
            raise ImportError("sdv is required for TVAEGenerator. Install with: pip install sdv")

        self._real_data = real_data.copy()
        self._all_columns = real_data.columns.tolist()
        self._categorical_columns = categorical_columns or []
        self._ordinal_columns = ordinal_columns or {}
        self._continuous_columns = continuous_columns or []

        # High-cardinality categoricals: top-k + OTHER, modelled jointly (see
        # ``collapse_high_cardinality``; they used to be refilled with real values).
        train_data, self._collapsed_cols = collapse_high_cardinality(
            real_data.copy(),
            self._categorical_columns,
            threshold=self.high_cardinality_threshold,
        )
        kept_cat_cols = list(self._categorical_columns)
        self._dropped_cols: list[str] = []
        ordinal_kept = {
            k: v for k, v in self._ordinal_columns.items() if k not in self._dropped_cols
        }

        # Create metadata
        metadata = _sdv.metadata_for(
            train_data,
            categorical=list(kept_cat_cols) + list(ordinal_kept),
            numerical=self._continuous_columns,
        )

        self._model = TVAESynthesizer(
            metadata=metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            enable_gpu=False,
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
            _sdv.fit(self._model, train_data)

        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using TVAE."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        if self.random_state is not None:
            np.random.seed(self.random_state)
            try:
                import torch

                torch.manual_seed(self.random_state)
            except ImportError:
                pass

        synthetic = _sdv.sample(self._model, n_samples)

        # Ensure column order matches original data
        final_cols = [c for c in self._all_columns if c in synthetic.columns]
        return synthetic[final_cols]
