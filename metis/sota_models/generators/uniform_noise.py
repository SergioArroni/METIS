"""Uniform noise baseline generator."""

import pandas as pd

from .base import BaseGenerator


class UniformNoiseGenerator(BaseGenerator):
    """
    Uniform noise generator - worst case baseline.

    Delegates to ``metis.calibrate.utils.noise_generator.UniformNoiseGenerator``
    for the actual generation logic, wrapping it in the BaseGenerator
    fit/generate interface used by the benchmarking pipeline.

    This represents the absolute worst baseline - purely random noise.
    Any reasonable generative model should significantly outperform this.
    """

    def __init__(
        self,
        name: str = "Uniform Noise",
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
        """Store real data and full column typing for generation."""
        self._real_data = real_data.copy()
        self._continuous_columns = continuous_columns or []
        self._categorical_columns = categorical_columns or []
        self._ordinal_columns = ordinal_columns or {}
        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate uniform noise via the canonical calibrate implementation."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        from metis.calibrate.utils.noise_generator import UniformNoiseGenerator as _CoreGenerator

        schema: dict[str, str] = {}
        for c in self._continuous_columns:
            schema[c] = "continuous"
        for c in self._categorical_columns:
            schema[c] = "categorical"
        for c in self._ordinal_columns:
            schema[c] = "ordinal"

        seed = self.random_state if self.random_state is not None else 0
        return _CoreGenerator(schema=schema).generate(self._real_data, n_samples, seed)
        return _CoreGenerator(schema=schema).generate(self._real_data, n_samples, seed)
