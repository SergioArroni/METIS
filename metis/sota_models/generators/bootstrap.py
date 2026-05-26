"""Bootstrap (random sampling) synthetic data generator."""

import pandas as pd

from .base import BaseGenerator


class RandomSamplingGenerator(BaseGenerator):
    """
    Bootstrap generator.

    Generates synthetic data by randomly sampling with replacement from the real data.
    This is the simplest baseline - tests whether generative models add value
    beyond simple resampling.
    """

    def __init__(
        self,
        name: str = "Bootstrap",
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.random_state = random_state
        self._real_data = None

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Store real data for sampling."""
        self._real_data = real_data.copy()
        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data by random sampling with replacement."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        return self._real_data.sample(
            n=n_samples, replace=True, random_state=self.random_state
        ).reset_index(drop=True)
