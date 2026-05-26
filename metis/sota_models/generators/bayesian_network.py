"""Bayesian Network synthetic data generator."""

import warnings

import numpy as np
import pandas as pd

from .base import BaseGenerator

# Optional imports
try:
    from pgmpy.estimators import BayesianEstimator, HillClimbSearch

    try:
        from pgmpy.models import DiscreteBayesianNetwork as PgmpyBayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork as PgmpyBayesianNetwork
    from pgmpy.sampling import BayesianModelSampling

    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    PgmpyBayesianNetwork = None


class BayesianNetworkGenerator(BaseGenerator):
    """
    Bayesian Network generator.

    Uses probabilistic graphical models to learn dependencies between variables
    and generate synthetic data that preserves these relationships.

    Requires: pgmpy
    """

    def __init__(
        self,
        name: str = "Bayesian Network",
        max_parents: int = 3,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.max_parents = max_parents
        self.random_state = random_state
        self._model = None
        self._columns = None

    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """Fit Bayesian Network to real data."""
        if not PGMPY_AVAILABLE:
            raise ImportError(
                "pgmpy is required for BayesianNetworkGenerator. Install with: pip install pgmpy"
            )

        self._real_data = real_data.copy()
        self._columns = real_data.columns.tolist()
        self._categorical_columns = categorical_columns or []
        self._ordinal_columns = ordinal_columns or {}
        self._continuous_columns = continuous_columns or []

        # Detect and exclude constant columns (nunique <= 1)
        # BN cannot compute conditional probabilities for constant nodes
        self._constant_columns = {
            col: real_data[col].iloc[0]
            for col in real_data.columns
            if real_data[col].nunique() <= 1
        }
        if self._constant_columns:
            warnings.warn(
                f"BN: excluding constant columns from model: {list(self._constant_columns.keys())}",
                stacklevel=2,
            )

        # Clean data: handle blanks/NaN in continuous columns
        data_discrete = real_data.drop(columns=list(self._constant_columns.keys())).copy()
        for col in data_discrete.columns:
            if data_discrete[col].dtype == object or pd.api.types.is_string_dtype(
                data_discrete[col]
            ):
                data_discrete[col] = (
                    data_discrete[col].astype(str).replace(r"^\s*$", np.nan, regex=True)
                )

        # Discretize continuous variables for BN
        self._discretize_info = {}

        for col in self._continuous_columns:
            if col in self._constant_columns:
                continue
            data_discrete[col] = pd.to_numeric(data_discrete[col], errors="coerce")
            data_discrete[col] = data_discrete[col].fillna(data_discrete[col].median())
            data_discrete[col], bins = pd.qcut(
                data_discrete[col], q=5, labels=False, duplicates="drop", retbins=True
            )
            self._discretize_info[col] = bins

        # Fill NaN in categorical columns with mode
        all_cat_cols = set(self._categorical_columns) | set(self._ordinal_columns.keys())
        for col in data_discrete.columns:
            if col in all_cat_cols or (
                (
                    data_discrete[col].dtype == object
                    or pd.api.types.is_string_dtype(data_discrete[col])
                )
                and col not in self._continuous_columns
            ):
                data_discrete[col] = data_discrete[col].fillna(
                    data_discrete[col].mode().iloc[0]
                    if len(data_discrete[col].mode()) > 0
                    else "missing"
                )

        # Convert non-numeric columns to category dtype for pgmpy compatibility
        for col in data_discrete.columns:
            if col not in self._continuous_columns and not pd.api.types.is_numeric_dtype(
                data_discrete[col]
            ):
                data_discrete[col] = data_discrete[col].astype(str).astype("category")

        # Learn structure
        hc = HillClimbSearch(data_discrete)
        best_model = hc.estimate(max_indegree=self.max_parents)

        # Learn parameters
        self._model = PgmpyBayesianNetwork(best_model.edges())
        self._model.fit(data_discrete, estimator=BayesianEstimator)

        self._sampler = BayesianModelSampling(self._model)
        self._is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data from Bayesian Network."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating data")

        synthetic = self._sampler.forward_sample(size=n_samples, seed=self.random_state)

        # Re-add constant columns with their fixed values
        for col, value in self._constant_columns.items():
            synthetic[col] = value

        # Reverse discretization for continuous variables
        for col in self._continuous_columns:
            if col in self._discretize_info and col in synthetic.columns:
                bins = self._discretize_info[col]
                synthetic[col] = synthetic[col].apply(
                    lambda x, bins=bins: (
                        np.random.uniform(bins[int(x)], bins[int(x) + 1])
                        if pd.notna(x) and int(x) < len(bins) - 1
                        else (bins[int(x)] if pd.notna(x) else bins[0])
                    )
                )

        # Reindex to match original column order, filling missing columns
        missing_cols = [c for c in self._columns if c not in synthetic.columns]
        if missing_cols:
            warnings.warn(
                f"BN forward_sample missing columns: {missing_cols}. "
                f"Filling from real data distribution.",
                stacklevel=2,
            )
            for col in missing_cols:
                synthetic[col] = (
                    self._real_data[col]
                    .sample(n=n_samples, replace=True, random_state=self.random_state)
                    .values
                )

        return synthetic[self._columns]
