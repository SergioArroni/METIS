"""Delete-and-impute baseline synthetic data generators."""

import numpy as np
import pandas as pd

from .base import BaseGenerator


class DeleteImputeGenerator(BaseGenerator):
    """
    Delete and impute baseline generator.

    Generates synthetic data by:
    1. Copying real data
    2. Randomly deleting a percentage of values
    3. Imputing deleted values with a simple strategy (0, mean, median, mode)
    """

    def __init__(
        self,
        name: str = "Delete-Impute",
        deletion_rate: float = 0.1,
        impute_strategy: str = "mean",
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.deletion_rate = deletion_rate
        self.impute_strategy = impute_strategy
        self.random_state = random_state
        self._real_data = None
        self._impute_values = {}

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Calculate imputation values from real data."""
        self._real_data = real_data.copy()
        self._categorical_columns = categorical_columns or []
        self._ordinal_columns = ordinal_columns or {}
        self._continuous_columns = continuous_columns or []

        for col in real_data.columns:
            if self.impute_strategy == "zero":
                self._impute_values[col] = 0
            elif self.impute_strategy == "mean":
                if col in self._categorical_columns or col in self._ordinal_columns:
                    self._impute_values[col] = real_data[col].mode()[0]
                elif pd.api.types.is_numeric_dtype(real_data[col]):
                    self._impute_values[col] = real_data[col].mean()
                else:
                    self._impute_values[col] = real_data[col].mode()[0]
            elif self.impute_strategy == "median":
                if col in self._categorical_columns:
                    self._impute_values[col] = real_data[col].mode()[0]
                elif pd.api.types.is_numeric_dtype(real_data[col]):
                    self._impute_values[col] = real_data[col].median()
                else:
                    self._impute_values[col] = real_data[col].mode()[0]
            elif self.impute_strategy == "mode":
                self._impute_values[col] = real_data[col].mode()[0]
            else:
                raise ValueError(f"Unknown impute_strategy: {self.impute_strategy}")

        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data by deleting and imputing values."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        synthetic = (
            self._real_data.sample(n=n_samples, replace=True, random_state=self.random_state)
            .reset_index(drop=True)
            .copy()
        )

        rng = np.random.RandomState(self.random_state)
        mask = rng.random(synthetic.shape) < self.deletion_rate
        synthetic_masked = synthetic.mask(mask)

        for col in synthetic.columns:
            original_dtype = synthetic[col].dtype
            synthetic_masked[col] = synthetic_masked[col].fillna(self._impute_values[col])
            if col in self._categorical_columns or col in self._ordinal_columns:
                synthetic_masked[col] = synthetic_masked[col].astype(original_dtype)

        return synthetic_masked


class DeleteImputeZeroGenerator(DeleteImputeGenerator):
    """Delete and impute with zero."""

    def __init__(
        self,
        name: str = "Delete-Impute-Zero",
        deletion_rate: float = 0.1,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            deletion_rate=deletion_rate,
            impute_strategy="zero",
            random_state=random_state,
            **kwargs,
        )


class DeleteImputeMeanGenerator(DeleteImputeGenerator):
    """Delete and impute with mean/mode."""

    def __init__(
        self,
        name: str = "Delete-Impute-Mean",
        deletion_rate: float = 0.1,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            deletion_rate=deletion_rate,
            impute_strategy="mean",
            random_state=random_state,
            **kwargs,
        )
