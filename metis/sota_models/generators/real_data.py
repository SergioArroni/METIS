"""Real data (split-half) upper bound baseline generator."""

import numpy as np
import pandas as pd

from .base import BaseGenerator


class RealDataGenerator(BaseGenerator):
    """
    Real data generator - upper bound baseline using split-half.

    Mirrors the calibration upper-bound strategy: splits the real dataset
    into two disjoint halves (without replacement) and uses one half as the
    "real" reference and the other as the "synthetic" output.  This ensures
    the benchmark upper bound is consistent with the calibration upper bound.

    The ``real_reference`` property exposes the "real" half so the benchmark
    orchestrator can use it instead of the full dataset when evaluating this
    generator.
    """

    def __init__(
        self,
        name: str = "Real Data",
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.random_state = random_state
        self._real_data = None
        self._real_half: pd.DataFrame | None = None
        self._synth_half: pd.DataFrame | None = None

    @property
    def real_reference(self) -> pd.DataFrame | None:
        """Return the 'real' half to be used as reference during evaluation."""
        return self._real_half

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Store real data."""
        self._real_data = real_data.copy()
        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate data using split-half (consistent with calibration upper bound).

        Shuffles the real data with the current seed and splits it into two
        disjoint halves.  Returns one half as "synthetic" data and stores
        the other in ``real_reference`` for use as the evaluation baseline.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        rng = np.random.RandomState(self.random_state)
        n = len(self._real_data)
        indices = rng.permutation(n)
        half = n // 2

        idx_a = indices[:half]
        idx_b = indices[half : half + half]

        self._real_half = self._real_data.iloc[idx_a].reset_index(drop=True)
        self._synth_half = self._real_data.iloc[idx_b].reset_index(drop=True)

        return self._synth_half.copy()
