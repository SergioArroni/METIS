"""CART synthesiser as in synthpop (Reiter, 2005; Nowok, Raab & Dibben, 2016).

The previous version bootstrapped real rows and then, three times over,
replaced every column by ``model.predict`` — the mean or mode of a leaf,
deterministically.  Without sampling, the variance vanished (a continuous
column came out with 3 distinct values, another with standard deviation 0).

Sequential CART, as synthpop's ``syn.cart``:

1. The first column is drawn from its real marginal (a random real value).
2. Each following column *j* has a tree fitted on columns ``1..j-1`` of the
   real data.  For each synthetic row, the row is dropped down the tree with
   its already-synthesised columns and the value of column *j* is a random
   **donor** among the real rows of that leaf (not the leaf's mean/mode), so
   the conditional distribution is sampled, not averaged.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import BaseGenerator


class CARTGenerator(BaseGenerator):
    """Sequential CART with donor sampling in the leaves (module docstring)."""

    def __init__(
        self,
        name: str = "CART",
        max_depth: int | None = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        random_state: int | None = None,
        **kwargs,
    ):
        """
        Args:
            max_depth: Tree depth limit (None: grown until ``min_samples_leaf``,
                as synthpop's default ``minbucket = 5``).
            min_samples_split: Minimum rows to split a node.
            min_samples_leaf: Minimum donors per leaf (synthpop ``minbucket``).
        """
        super().__init__(name=name, **kwargs)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._models: dict[str, object] = {}
        self._columns: list[str] = []

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Fit one tree per column on the columns before it (visit order = column order)."""
        self._real = real_data.reset_index(drop=True).copy()
        self._columns = real_data.columns.tolist()
        cat = set(categorical_columns or []) | set((ordinal_columns or {}).keys())
        self._categorical = {
            c for c in self._columns if c in cat or not pd.api.types.is_numeric_dtype(real_data[c])
        }
        self._encoded = self._encode(self._real)

        self._models = {}
        self._leaves: dict[str, dict[int, np.ndarray]] = {}
        for j, col in enumerate(self._columns[1:], start=1):
            features = self._columns[:j]
            tree_cls = DecisionTreeClassifier if col in self._categorical else DecisionTreeRegressor
            tree = tree_cls(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            y = self._encoded[col]
            tree.fit(self._encoded[features], y)
            leaves = tree.apply(self._encoded[features])
            self._models[col] = tree
            # Real row indices (donors) of every leaf.
            self._leaves[col] = {
                int(leaf): np.flatnonzero(leaves == leaf) for leaf in np.unique(leaves)
            }
        self._is_fitted = True

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Numeric matrix for the trees: categorical as integer codes, NaN as a code / median."""
        out = pd.DataFrame(index=frame.index)
        self._codes = getattr(self, "_codes", {})
        for col in self._columns:
            if col in self._categorical:
                values = frame[col].astype(str).where(frame[col].notna(), "__missing__")
                if col not in self._codes:
                    self._codes[col] = {v: i for i, v in enumerate(sorted(values.unique()))}
                out[col] = values.map(self._codes[col]).fillna(-1).astype(float)
            else:
                v = pd.to_numeric(frame[col], errors="coerce")
                out[col] = v.fillna(self._real[col].median() if col in self._real else 0.0)
        return out

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Synthesise column by column, sampling a donor in each leaf."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")
        rng = np.random.default_rng(self.random_state)
        first = self._columns[0]
        donors = rng.integers(len(self._real), size=n_samples)
        synth = pd.DataFrame({first: self._real[first].to_numpy()[donors]})
        encoded = pd.DataFrame({first: self._encoded[first].to_numpy()[donors]})

        for j, col in enumerate(self._columns[1:], start=1):
            features = self._columns[:j]
            leaves = self._models[col].apply(encoded[features])
            picked = np.empty(n_samples, dtype=int)
            for leaf in np.unique(leaves):
                rows = np.flatnonzero(leaves == leaf)
                pool = self._leaves[col][int(leaf)]
                picked[rows] = pool[rng.integers(len(pool), size=len(rows))]
            synth[col] = self._real[col].to_numpy()[picked]
            encoded[col] = self._encoded[col].to_numpy()[picked]

        return synth[self._columns]
