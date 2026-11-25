"""SMOTE-NC synthetic data generator.

The previous version called ``imblearn``'s ``SMOTENC.fit_resample`` with
``sampling_strategy="auto"``: that *oversamples the minority classes* and
returns the original rows plus a few interpolated ones, so the "synthetic"
table was mostly real rows (94 % with a balanced target), the target was
forced to 50/50 and, in regression, every feature row was a real copy with
the target re-drawn inside its decile.  It was a bootstrap, not a generator.

This version uses the SMOTE-NC rule (Chawla et al., 2002) as a synthesiser:
every output row is new.

1. Draw a class with the real class frequencies (classification) — the
   target marginal is kept; in regression the target is one more continuous
   column and there are no classes.
2. Draw a seed row of that class and one of its ``k`` nearest neighbours of
   the same class (distance on standardised continuous columns plus, for
   each categorical mismatch, the median of the continuous standard
   deviations, as in SMOTE-NC).
3. Continuous columns: ``x_seed + u · (x_neighbour − x_seed)``, ``u ~ U(0, 1)``
   (integer-valued columns are rounded).  Categorical and ordinal columns: the
   most frequent value among the ``k`` neighbours (SMOTE-NC's vote).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base import BaseGenerator


class SMOTENCGenerator(BaseGenerator):
    """SMOTE-NC interpolation as a synthesiser (module docstring)."""

    def __init__(
        self,
        name: str = "SMOTENC",
        *,
        k_neighbors: int = 5,
        random_state: int | None = None,
        target_column: str | None = None,
        task_type: str | None = None,
        n_quantile_bins: int = 10,  # kept for config compatibility; unused
        schema_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.target_column = target_column
        self.task_type = task_type
        self.n_quantile_bins = n_quantile_bins
        self.schema_config = schema_config
        self._data: pd.DataFrame | None = None

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Store the complete rows and split continuous from categorical columns."""
        if not self.target_column:
            raise ValueError(
                "target_column is required for SMOTENC. Specify target_column in generator params."
            )
        if self.target_column not in real_data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in data. "
                f"Available columns: {list(real_data.columns)}"
            )
        # Interpolation needs complete rows (as SMOTE-NC itself).
        data = real_data.dropna().reset_index(drop=True)
        if len(data) < 2:
            raise ValueError("SMOTENC needs at least two complete rows")

        cat = set(categorical_columns or []) | set((ordinal_columns or {}).keys())
        self._continuous = []
        self._categorical = []
        for col in data.columns:
            if col in cat:
                self._categorical.append(col)
                continue
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().all() and col not in cat:
                data[col] = converted
                self._continuous.append(col)
            else:
                self._categorical.append(col)

        if self.task_type is None:
            y = data[self.target_column]
            self.task_type = (
                "classification"
                if self.target_column in self._categorical or y.nunique() <= 20
                else "regression"
            )
        if self.task_type == "classification" and self.target_column in self._continuous:
            # Classes are never interpolated.
            self._continuous.remove(self.target_column)
            self._categorical.append(self.target_column)

        self._integer = {
            c
            for c in self._continuous
            if np.allclose(data[c] % 1, 0) and data[c].dtype.kind in "if"
        }
        self._data = data
        self._is_fitted = True

    # ------------------------------------------------------------------
    def _space(self, frame: pd.DataFrame) -> np.ndarray:
        """Mixed distance space: z-scored continuous + scaled one-hot categorical."""
        blocks = []
        stds = []
        for c in self._continuous:
            v = frame[c].to_numpy(float)
            sd = v.std() or 1.0
            stds.append(sd)
            blocks.append(((v - v.mean()) / sd)[:, None])
        # SMOTE-NC: a categorical mismatch costs the median continuous std
        # (1 after z-scoring, so the penalty is 1 when there are continuous columns).
        penalty = 1.0
        for c in self._categorical:
            if c == self.target_column and self.task_type == "classification":
                continue
            codes = pd.get_dummies(frame[c].astype(str)).to_numpy(float)
            blocks.append(codes * penalty / np.sqrt(2.0))
        if not blocks:
            return np.zeros((len(frame), 1))
        return np.hstack(blocks)

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate ``n_samples`` new rows by SMOTE-NC interpolation."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")
        rng = np.random.default_rng(self.random_state)
        data = self._data
        space = self._space(data)

        if self.task_type == "classification":
            labels = data[self.target_column].astype(str).to_numpy()
            classes, counts = np.unique(labels, return_counts=True)
            groups = [np.flatnonzero(labels == c) for c in classes]
            drawn = rng.choice(len(classes), size=n_samples, p=counts / counts.sum())
        else:
            groups = [np.arange(len(data))]
            drawn = np.zeros(n_samples, dtype=int)

        rows = []
        for g, members in enumerate(groups):
            want = int(np.sum(drawn == g))
            if want == 0:
                continue
            block = data.iloc[members].reset_index(drop=True)
            if len(members) == 1:
                # A class with one row: nothing to interpolate with.
                rows.append(block.iloc[np.zeros(want, dtype=int)])
                continue
            k = min(self.k_neighbors, len(members) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(space[members])
            neighbours = nn.kneighbors(space[members], return_distance=False)[:, 1:]
            seeds = rng.integers(len(members), size=want)
            mates = neighbours[seeds, rng.integers(k, size=want)]
            out = pd.DataFrame(index=range(want))
            u = rng.random(want)
            for c in self._continuous:
                a = block[c].to_numpy(float)[seeds]
                b = block[c].to_numpy(float)[mates]
                v = a + u * (b - a)
                out[c] = np.round(v) if c in self._integer else v
            for c in self._categorical:
                values = block[c].to_numpy()
                votes = values[neighbours[seeds]]  # k neighbours of each seed
                out[c] = [pd.Series(v).mode().iloc[0] for v in votes]
            if self.task_type == "classification":
                out[self.target_column] = block[self.target_column].iloc[0]
            rows.append(out)

        synthetic = pd.concat(rows, ignore_index=True)[list(data.columns)]
        order = rng.permutation(len(synthetic))
        return synthetic.iloc[order].reset_index(drop=True)
