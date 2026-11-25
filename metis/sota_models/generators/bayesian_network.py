"""Bayesian Network synthetic data generator."""

import warnings

import numpy as np
import pandas as pd

from .base import BaseGenerator

# Optional imports
try:
    from pgmpy.causal_discovery import HillClimbSearch
    from pgmpy.estimators import BayesianEstimator

    try:
        from pgmpy.models import DiscreteBayesianNetwork as PgmpyBayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork as PgmpyBayesianNetwork
    from pgmpy.sampling import BayesianModelSampling

    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    PgmpyBayesianNetwork = None


def _bayesian_estimator():
    """Bayesian (BDeu) parameter estimator for the installed pgmpy.

    pgmpy >= 1.1 wants an *instance* of a discrete estimator and raised
    ``TypeError`` with the class ``BayesianEstimator`` the generator passed, so
    BN failed on any recent install; older pgmpy wants the class.
    """
    try:
        from pgmpy.parameter_estimator import DiscreteBayesianEstimator
    except ImportError:
        return BayesianEstimator
    return DiscreteBayesianEstimator()


# BDeu equivalent sample size, pgmpy's default for BayesianEstimator.
_BDEU_EQUIVALENT_SAMPLE_SIZE = 5


def _ensure_isolated_cpds(model, data: pd.DataFrame) -> None:
    """Give every node without a CPD its marginal (BDeu-smoothed) distribution.

    pgmpy 1.0 ``fit`` creates no CPD for a node without edges, and sampling
    then fails with "No CPD associated with <node>" (the AWS install); pgmpy
    1.1 does create it.  The CPD is the one BayesianEstimator would give a
    parentless node: counts plus ``ess / K`` pseudo-counts per state.
    """
    from pgmpy.factors.discrete import TabularCPD  # noqa: PLC0415

    for node in model.nodes():
        if model.get_cpds(node) is not None:
            continue
        counts = data[node].value_counts(sort=False)
        states = list(counts.index)
        alpha = _BDEU_EQUIVALENT_SAMPLE_SIZE / len(states)
        probs = (counts.to_numpy(float) + alpha) / (counts.sum() + alpha * len(states))
        model.add_cpds(
            TabularCPD(
                variable=node,
                variable_card=len(states),
                values=[[float(p)] for p in probs],
                state_names={node: states},
            )
        )


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

        # Real values of every bin: a synthetic bin is turned back into a real
        # value drawn from that bin (an empirical donor), not a uniform draw
        # between the bin edges, which with heavy tails put half of the top
        # quintile near the maximum (Airbnb price).
        self._bin_values: dict[str, dict[int, np.ndarray]] = {}
        for col in self._continuous_columns:
            if col in self._constant_columns:
                continue
            numeric = pd.to_numeric(data_discrete[col], errors="coerce")
            observed = numeric.dropna()
            data_discrete[col] = numeric.fillna(observed.median())
            data_discrete[col], bins = pd.qcut(
                data_discrete[col], q=5, labels=False, duplicates="drop", retbins=True
            )
            self._discretize_info[col] = bins
            codes = data_discrete[col].to_numpy()
            values = numeric.to_numpy(float)
            self._bin_values[col] = {
                int(b): values[(codes == b) & ~np.isnan(values)] for b in np.unique(codes)
            }

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
        hc = HillClimbSearch(max_indegree=self.max_parents, return_type="dag", show_progress=False)
        best_model = hc.fit(data_discrete).causal_graph_

        # Learn parameters.  Isolated nodes too: built from the edges only, a
        # column with no edge vanished from the network and was refilled with a
        # bootstrap of the real column.
        self._model = PgmpyBayesianNetwork(best_model.edges())
        self._model.add_nodes_from(data_discrete.columns)
        self._model.fit(data_discrete, estimator=_bayesian_estimator())
        _ensure_isolated_cpds(self._model, data_discrete)

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

        # Reverse discretization for continuous variables.  A generator-owned
        # RNG: the global ``np.random`` made these columns unreproducible.
        rng = np.random.default_rng(self.random_state)
        for col in self._continuous_columns:
            if col in self._bin_values and col in synthetic.columns:
                pools = self._bin_values[col]
                codes = pd.to_numeric(synthetic[col], errors="coerce").to_numpy()
                out = np.full(len(codes), np.nan)
                for b, pool in pools.items():
                    rows = np.flatnonzero(codes == b)
                    if len(rows) and len(pool):
                        out[rows] = pool[rng.integers(len(pool), size=len(rows))]
                synthetic[col] = out

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
