"""Real data baseline: real rows the generator pipeline never saw.

The benchmark splits the real table per seed into train / holdout / unseen
(``comparison.orchestrator.split_real``).  ``real_data`` returns rows of the
*unseen* part: it is what a perfect generator that never memorised anything
would produce, so it is the reference of the best case in fidelity, utility
**and** privacy.  It used to split the whole table in two halves and be
evaluated as n/2 vs n/2 while every other generator was n vs n; and the
paper used it as the *lower* bound of privacy, which an unseen sample is not.
"""

import numpy as np
import pandas as pd

from .base import BaseGenerator


class RealDataGenerator(BaseGenerator):
    """Returns rows of the unseen real part (module docstring)."""

    def __init__(
        self,
        name: str = "Real Data",
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.random_state = random_state
        self._unseen: pd.DataFrame | None = None

    def set_unseen(self, unseen: pd.DataFrame) -> None:
        """Real rows outside the training part and the holdout."""
        self._unseen = unseen.reset_index(drop=True)

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Nothing to learn; without ``set_unseen`` it refuses to copy *real_data*."""
        if self._unseen is None:
            raise ValueError(
                "real_data needs real rows the evaluation never sees: call set_unseen() "
                "(the benchmark orchestrator does); returning the training rows would be a copy"
            )
        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """``n_samples`` unseen real rows, without replacement when there are enough."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")
        rng = np.random.default_rng(self.random_state)
        n = len(self._unseen)
        idx = rng.choice(n, size=n_samples, replace=n_samples > n)
        return self._unseen.iloc[idx].reset_index(drop=True)
