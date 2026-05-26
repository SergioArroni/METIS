"""CART (Classification and Regression Trees) synthetic data generator."""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import BaseGenerator


class CARTGenerator(BaseGenerator):
    """
    CART (Classification and Regression Trees) generator.

    Uses decision trees to model conditional distributions and generate
    synthetic data by traversing the learned tree structure.
    """

    def __init__(
        self,
        name: str = "CART",
        max_depth: int = 10,
        min_samples_split: int = 10,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self._models = {}
        self._columns = None

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Fit CART models to real data."""
        from sklearn.preprocessing import LabelEncoder

        self._real_data = real_data.copy()
        self._columns = real_data.columns.tolist()
        self._categorical_columns = categorical_columns or []
        self._ordinal_columns = ordinal_columns or {}
        self._continuous_columns = continuous_columns or []

        all_cat_cols = set(self._categorical_columns) | set(self._ordinal_columns.keys())

        # Label-encode all categorical columns for sklearn
        self._label_encoders = {}
        data_encoded = real_data.copy()
        for col in self._columns:
            if col in all_cat_cols or data_encoded[col].dtype == object:
                le = LabelEncoder()
                data_encoded[col] = data_encoded[col].fillna("__missing__").astype(str)
                data_encoded[col] = le.fit_transform(data_encoded[col])
                self._label_encoders[col] = le
            else:
                data_encoded[col] = pd.to_numeric(data_encoded[col], errors="coerce")
                data_encoded[col] = data_encoded[col].fillna(data_encoded[col].median())

        self._data_encoded = data_encoded

        # Train a tree for each column using others as features
        for target_col in self._columns:
            feature_cols = [c for c in self._columns if c != target_col]
            X = data_encoded[feature_cols]
            y = data_encoded[target_col]

            if target_col in all_cat_cols or target_col in self._label_encoders:
                model = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )

            model.fit(X, y)
            self._models[target_col] = model

        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using CART models."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        synthetic = (
            self._data_encoded.sample(n=n_samples, replace=True, random_state=self.random_state)
            .reset_index(drop=True)
            .copy()
        )

        # Iteratively refine each column using trained trees
        for _ in range(3):
            for target_col in self._columns:
                feature_cols = [c for c in self._columns if c != target_col]
                X = synthetic[feature_cols]
                model = self._models[target_col]
                synthetic[target_col] = model.predict(X)

        # Decode categorical columns back to original labels
        result = synthetic.copy()
        for col, le in self._label_encoders.items():
            if col in result.columns:
                vals = result[col].round().astype(int).clip(0, len(le.classes_) - 1)
                result[col] = le.inverse_transform(vals)

        return result
