"""SMOTENC synthetic data generator."""

import numpy as np
import pandas as pd

from .base import BaseGenerator

# Optional import for SMOTE and SMOTENC
try:
    from imblearn.over_sampling import SMOTE, SMOTENC

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    SMOTE = None
    SMOTENC = None

# Optional import for METIS preprocessing (SimpleCaster)
try:
    from metis.domain.contracts import TypeSchema
    from metis.infrastructure.preprocess.caster import SimpleCaster

    METIS_CASTER_AVAILABLE = True
except ImportError:
    METIS_CASTER_AVAILABLE = False
    TypeSchema = None
    SimpleCaster = None


class SMOTENCGenerator(BaseGenerator):
    """
    SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous) generator.

    Generates synthetic samples by interpolating between existing samples,
    natively handling both categorical and continuous features.
    Uses the same preprocessing as METIS evaluation (SimpleCaster + dropna):
    1. Drop rows with NaN (same as METIS orchestrator)
    2. LabelEncode categoricals for SMOTENC
    3. For regression targets: bin into quantiles to create pseudo-classes
    4. Run SMOTENC (falls back to plain SMOTE if no categorical columns)
    5. Reverse transforms to produce raw output
    """

    def __init__(
        self,
        name: str = "SMOTENC",
        k_neighbors: int = 5,
        random_state: int | None = None,
        target_column: str | None = None,
        task_type: str | None = None,
        n_quantile_bins: int = 10,
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
        self._real_data = None
        self._target = None
        self._target_bin_edges = None
        self._caster = None

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Fit SMOTENC to real data with METIS-consistent preprocessing."""
        if not SMOTE_AVAILABLE:
            raise ImportError(
                "imbalanced-learn is required for SMOTENCGenerator. "
                "Install with: pip install imbalanced-learn>=0.12.0"
            )

        if not self.target_column:
            raise ValueError(
                "target_column is required for SMOTENC. Specify target_column in generator params."
            )

        if self.target_column not in real_data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in data. "
                f"Available columns: {list(real_data.columns)}"
            )

        # Step 1: Drop NaN rows (same as METIS orchestrator)
        n_before = len(real_data)
        clean_data = real_data.dropna().reset_index(drop=True)
        n_dropped = n_before - len(clean_data)
        if n_dropped > 0:
            print(
                f"  SMOTENC: dropped {n_dropped} rows with NaN ({n_dropped / n_before * 100:.1f}%)"
            )

        # Separate features and target
        self._target = clean_data[self.target_column].copy()
        self._real_data = clean_data.drop(columns=[self.target_column]).copy()

        self._categorical_columns = categorical_columns or []
        self._ordinal_columns = ordinal_columns or {}
        self._continuous_columns = continuous_columns or []

        # Auto-detect task type if not specified
        if self.task_type is None:
            n_unique = self._target.nunique()
            if n_unique <= 20 or self._target.dtype == object:
                self.task_type = "classification"
            else:
                self.task_type = "regression"

        self._is_fitted = True

    def _preprocess(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Preprocess data for SMOTENC.

        Returns:
            tuple of (X_clean, y_clean, cat_indices, label_encoders)
        """
        from sklearn.preprocessing import LabelEncoder

        X_clean = X.copy()
        y_clean = y.copy()

        categorical_cols = set(self._categorical_columns) | set(self._ordinal_columns.keys())

        # Fix mistyped columns: numeric columns stored as strings
        for col in X_clean.columns:
            if col not in categorical_cols and X_clean[col].dtype == object:
                X_clean[col] = X_clean[col].replace(r"^\s*$", np.nan, regex=True)
                converted = pd.to_numeric(X_clean[col], errors="coerce")
                if converted.notna().any():
                    X_clean[col] = converted
                else:
                    categorical_cols.add(col)

        # Drop any rows that became NaN after type conversion
        mask = X_clean.notna().all(axis=1) & y_clean.notna()
        n_before = len(X_clean)
        X_clean = X_clean[mask].reset_index(drop=True)
        y_clean = y_clean[mask].reset_index(drop=True)
        n_dropped = n_before - len(X_clean)
        if n_dropped > 0:
            print(f"  SMOTENC: dropped {n_dropped} additional rows after type coercion")

        # LabelEncode categoricals for SMOTENC
        label_encoders = {}
        for col in X_clean.columns:
            if col in categorical_cols:
                le = LabelEncoder()
                X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                label_encoders[col] = le

        # Categorical feature indices for SMOTENC
        cat_indices = [i for i, col in enumerate(X_clean.columns) if col in categorical_cols]

        return X_clean, y_clean, cat_indices, label_encoders

    def _bin_regression_target(self, y: pd.Series) -> pd.Series:
        """Bin a continuous target into quantile-based classes for SMOTENC."""
        try:
            y_binned, bin_edges = pd.qcut(
                y, q=self.n_quantile_bins, labels=False, retbins=True, duplicates="drop"
            )
        except ValueError:
            y_binned, bin_edges = pd.cut(
                y,
                bins=min(self.n_quantile_bins, y.nunique()),
                labels=False,
                retbins=True,
            )
        self._target_bin_edges = bin_edges
        return y_binned.astype(int)

    def _unbin_regression_target(
        self, y_binned: pd.Series, rng: np.random.RandomState
    ) -> pd.Series:
        """Reconstruct continuous values from binned classes."""
        edges = self._target_bin_edges
        y_continuous = np.empty(len(y_binned))
        for i in range(len(y_binned)):
            bin_idx = int(y_binned.iloc[i])
            bin_idx = min(bin_idx, len(edges) - 2)
            low, high = edges[bin_idx], edges[bin_idx + 1]
            y_continuous[i] = rng.uniform(low, high)
        return pd.Series(y_continuous, index=y_binned.index)

    def _decode_categoricals(self, df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
        """Reverse label encoding for categorical columns."""
        result = df.copy()
        for col, le in label_encoders.items():
            if col in result.columns:
                vals = result[col].round().astype(int).clip(0, len(le.classes_) - 1)
                result[col] = le.inverse_transform(vals)
        return result

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using SMOTENC interpolation."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        rng = np.random.RandomState(self.random_state)
        X = self._real_data.copy()
        y = self._target.copy()

        X_clean, y_clean, cat_indices, label_encoders = self._preprocess(X, y)

        if self.task_type == "regression":
            y_for_smote = self._bin_regression_target(y_clean)
        else:
            y_for_smote = y_clean.copy()

        y_for_smote = y_for_smote.astype(str)

        class_counts = y_for_smote.value_counts()
        min_count = class_counts.min()
        k = min(self.k_neighbors, min_count - 1) if min_count > 1 else 1

        if cat_indices:
            smote = SMOTENC(
                categorical_features=cat_indices,
                sampling_strategy="auto",
                k_neighbors=k,
                random_state=self.random_state,
            )
        else:
            smote = SMOTE(
                sampling_strategy="auto",
                k_neighbors=k,
                random_state=self.random_state,
            )

        X_resampled, y_resampled = smote.fit_resample(X_clean, y_for_smote)

        synthetic = pd.DataFrame(X_resampled, columns=X.columns)

        if self.task_type == "regression":
            y_series = pd.Series(y_resampled, name=self.target_column).astype(int)
            synthetic[self.target_column] = self._unbin_regression_target(y_series, rng)
        else:
            synthetic[self.target_column] = y_resampled

        synthetic = self._decode_categoricals(synthetic, label_encoders)

        replace = len(synthetic) < n_samples
        return synthetic.sample(
            n=n_samples, replace=replace, random_state=self.random_state
        ).reset_index(drop=True)
