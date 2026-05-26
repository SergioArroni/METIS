"""Gaussian Copula baseline synthetic data generator."""

import warnings

import numpy as np
import pandas as pd

from .base import BaseGenerator

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False


class GaussianCopulaGenerator(BaseGenerator):
    """
    Gaussian Copula baseline generator.

    Models the marginal distributions of each column independently
    and captures linear inter-column dependencies via a Gaussian copula.
    This is a lightweight probabilistic baseline that sits between
    simple resampling and full deep-generative models.

    Requires: sdv (Synthetic Data Vault)
    """

    def __init__(
        self,
        name: str = "Gaussian Copula",
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.random_state = random_state
        self._model = None
        self._all_columns: list[str] = []

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Fit a Gaussian Copula model to the real data."""
        if not SDV_AVAILABLE:
            raise ImportError(
                "sdv is required for GaussianCopulaGenerator. Install with: pip install sdv"
            )

        self._all_columns = real_data.columns.tolist()
        categorical_columns = categorical_columns or []
        ordinal_columns = ordinal_columns or {}
        continuous_columns = continuous_columns or []

        # Build SDV metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_data)

        for col in categorical_columns:
            if col in real_data.columns:
                metadata.update_column(col, sdtype="categorical")
        for col in ordinal_columns:
            if col in real_data.columns:
                metadata.update_column(col, sdtype="categorical")
        for col in continuous_columns:
            if col in real_data.columns:
                metadata.update_column(col, sdtype="numerical")

        self._model = GaussianCopulaSynthesizer(metadata=metadata)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(real_data)

        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data from the fitted Gaussian Copula."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        synthetic = self._model.sample(num_rows=n_samples)

        # Ensure column order matches original data
        final_cols = [c for c in self._all_columns if c in synthetic.columns]
        return synthetic[final_cols]
