"""
Base utilities for metric implementations.

Provides ``MetricBase``, the root class for every METIS metric.  It
handles data binding, caching, and commonly needed statistical helpers
so that concrete metric classes can focus exclusively on their domain
logic.

Typical usage inside a concrete metric::

    class MyMetric(MetricBase):
        def fit(self, real, synth, ctx):
            self._setup(real, synth, ctx)
            return self

        def compute(self) -> MetricResult:
            nums = self._get_numeric_columns()
            ...
"""

import os
from typing import Any

import pandas as pd


def _env_int(name: str, default: int) -> int:
    """Parse an int from an env var, falling back to *default* on bad input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# Privacy-by-default knobs for descriptive-statistics emission. Override via
# environment variables so reports cannot be made more verbose accidentally
# from a YAML config:
#   - METIS_REDACT_MIN_COUNT: small-cell threshold; categories with a count
#     strictly below this value are aggregated into ``__redacted__``
#     (default 5, mirroring common k-anonymity guidance).
#   - METIS_REDACT_TOP_K: cap on the number of categories preserved verbatim;
#     the rest are aggregated into ``__other__`` (default 50). 0 disables.
_REDACT_MIN_COUNT = _env_int("METIS_REDACT_MIN_COUNT", 5)
_REDACT_TOP_K = _env_int("METIS_REDACT_TOP_K", 50)


def _redact_value_counts(value_counts: pd.Series) -> dict[Any, int]:
    """Return a privacy-preserving dict view of ``value_counts``.

    Two redactions are applied:

    1. **Small-cell suppression**: categories whose count is strictly
       below :data:`_REDACT_MIN_COUNT` are removed individually and their
       total is reported under the ``__redacted__`` key.
    2. **Top-K capping**: only the :data:`_REDACT_TOP_K` most frequent
       remaining categories are kept; everything else is aggregated under
       the ``__other__`` key.

    Both knobs honour environment variables (see module docstring) and can
    be disabled by setting the corresponding env var to ``0``.
    """
    if value_counts.empty:
        return {}

    counts = value_counts.sort_values(ascending=False)
    redacted_total = 0
    if _REDACT_MIN_COUNT > 1:
        small = counts[counts < _REDACT_MIN_COUNT]
        redacted_total = int(small.sum())
        counts = counts[counts >= _REDACT_MIN_COUNT]

    other_total = 0
    if _REDACT_TOP_K > 0 and len(counts) > _REDACT_TOP_K:
        other_total = int(counts.iloc[_REDACT_TOP_K:].sum())
        counts = counts.iloc[:_REDACT_TOP_K]

    out: dict[Any, int] = {k: int(v) for k, v in counts.items()}
    if other_total:
        out["__other__"] = other_total
    if redacted_total:
        out["__redacted__"] = redacted_total
    return out


class MetricBase:
    """Root base class for all METIS metrics.

    Provides:
    - Data binding via ``_setup()``
    - Column introspection (``_get_numeric_columns``, ``_get_categorical_columns``)
    - Transparent caching through the shared ``StatsStore``
    - Pre-built helpers for univariate stats, correlations, CV splits, and KNN

    Subclasses must call ``_setup()`` in their ``fit()`` method before
    accessing ``_real_data``, ``_synth_data``, or ``_context``.
    """

    def __init__(self) -> None:
        self._real_data: pd.DataFrame | None = None
        self._synth_data: pd.DataFrame | None = None
        self._context: dict[str, Any] = {}
        self._stats_store = None

    def _setup(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> None:
        """Bind data and context for subsequent computations.

        Args:
            real_data: Original (real) dataset.
            synth_data: Synthetic dataset to evaluate.
            context: Shared context dict containing at minimum:
                - ``stats_store``: optional ``StatsStore`` for caching.
                - ``seed``: random seed for reproducibility.
                - ``dataset_spec``: ``DatasetSpec`` with target / task_type info.
        """
        self._real_data = real_data
        self._synth_data = synth_data
        self._context = context
        self._stats_store = context.get("stats_store")

    # ------------------------------------------------------------------
    # Column introspection
    # ------------------------------------------------------------------

    def _get_numeric_columns(self, exclude_target: bool = True) -> list[str]:
        """Return numeric column names from the real dataset.

        Args:
            exclude_target: If ``True`` and a target column is set in the
                context, it will be excluded from the result.
        """
        numeric_cols = self._real_data.select_dtypes(include=["number"]).columns.tolist()

        if exclude_target and "target" in self._context:
            target = self._context["target"]
            if target in numeric_cols:
                numeric_cols.remove(target)

        return numeric_cols

    def _get_categorical_columns(self, exclude_target: bool = True) -> list[str]:
        """Return categorical column names from the real dataset.

        Args:
            exclude_target: If ``True`` and a target column is set in the
                context, it will be excluded from the result.
        """
        categorical_cols = self._real_data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if exclude_target and "target" in self._context:
            target = self._context["target"]
            if target in categorical_cols:
                categorical_cols.remove(target)

        return categorical_cols

    def _select_columns_by_type(self, column_type: str) -> list[str]:
        """Select columns by a high-level type label.

        Args:
            column_type: One of ``"numeric"``, ``"categorical"``, or ``"all"``.

        Returns:
            list of matching column names.

        Raises:
            ValueError: If *column_type* is not recognised.
        """
        if column_type == "numeric":
            return self._get_numeric_columns()
        if column_type == "categorical":
            return self._get_categorical_columns()
        if column_type == "all":
            return self._real_data.columns.tolist()
        raise ValueError(f"Unknown column type: {column_type}")

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _get_cached_or_compute(self, cache_key: str, compute_fn) -> Any:
        """Retrieve a value from the shared cache, computing it on a miss.

        If no ``StatsStore`` is available (e.g. in unit tests), the value
        is computed directly without caching.

        Args:
            cache_key: Unique string key for this computation.
            compute_fn: Zero-argument callable that produces the value.
        """
        if self._stats_store:
            return self._stats_store.get_or_compute(cache_key, compute_fn)
        return compute_fn()

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    def _get_univariate_stats(self, column: str, dataset: str = "real") -> dict[str, Any]:
        """Return cached descriptive statistics for a single column.

        For numeric columns: mean, std, min, max, median, Q25, Q75, skew,
        kurtosis.  For categorical: unique_count, mode, value_counts.

        Args:
            column: Column name to analyse.
            dataset: ``"real"`` or ``"synth"``.
        """
        data = self._real_data if dataset == "real" else self._synth_data
        cache_key = f"univariate:{dataset}:{column}"

        def compute_stats():
            series = data[column]

            if pd.api.types.is_numeric_dtype(series):
                return {
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "median": series.median(),
                    "q25": series.quantile(0.25),
                    "q75": series.quantile(0.75),
                    "skew": series.skew(),
                    "kurtosis": series.kurtosis(),
                }
            value_counts = series.value_counts()
            return {
                "unique_count": series.nunique(),
                "mode": series.mode().iloc[0] if len(series.mode()) > 0 else None,
                "most_frequent_value": (value_counts.index[0] if len(value_counts) > 0 else None),
                "most_frequent_count": (value_counts.iloc[0] if len(value_counts) > 0 else 0),
                "value_counts": _redact_value_counts(value_counts),
            }

        return self._get_cached_or_compute(cache_key, compute_stats)

    def _get_correlation_matrix(self, dataset: str = "real") -> pd.DataFrame:
        """Return the cached Pearson correlation matrix for numeric columns.

        Args:
            dataset: ``"real"`` or ``"synth"``.

        Returns:
            Correlation DataFrame, or an empty DataFrame if there are no
            numeric columns.
        """
        data = self._real_data if dataset == "real" else self._synth_data
        cache_key = f"correlation:{dataset}"

        def compute_correlation():
            numeric_data = data.select_dtypes(include=["number"])
            if numeric_data.empty:
                return pd.DataFrame()
            return numeric_data.corr()

        return self._get_cached_or_compute(cache_key, compute_correlation)

    def _get_train_test_splits(
        self, n_splits: int = 3, random_state: int = 42
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Return cached K-Fold train/test splits of the real dataset.

        Args:
            n_splits: Number of folds.
            random_state: Seed for the shuffle.

        Returns:
            list of ``(train_df, test_df)`` tuples.
        """
        cache_key = f"splits:n{n_splits}:seed{random_state}"

        def compute_splits():
            import numpy as np
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            indices = np.arange(len(self._real_data))

            return [
                (self._real_data.iloc[train_idx], self._real_data.iloc[test_idx])
                for train_idx, test_idx in kf.split(indices)
            ]

        return self._get_cached_or_compute(cache_key, compute_splits)

    def _get_knn_distances(self, k: int = 5) -> dict[str, Any]:
        """Return cached KNN distances from synthetic to real records.

        Numeric columns are standardised before computing distances so
        that all features contribute on a comparable scale.

        Args:
            k: Number of nearest neighbours.

        Returns:
            dict with keys ``distances``, ``indices``, ``mean_distance``,
            ``min_distance``, ``max_distance``, ``k``.  If no numeric
            columns are available, returns ``{"error": ...}``.
        """
        cache_key = f"knn:k{k}"

        def compute_knn():
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler

            numeric_cols = self._get_numeric_columns()
            if not numeric_cols:
                return {"error": "No numeric columns found for KNN computation"}

            real_numeric = self._real_data[numeric_cols].fillna(0)
            synth_numeric = self._synth_data[numeric_cols].fillna(0)

            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_numeric)
            synth_scaled = scaler.transform(synth_numeric)

            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(real_scaled)

            distances, indices = knn.kneighbors(synth_scaled)

            return {
                "distances": distances,
                "indices": indices,
                "mean_distance": float(np.mean(distances)),
                "min_distance": float(np.min(distances)),
                "max_distance": float(np.max(distances)),
                "k": k,
            }

        return self._get_cached_or_compute(cache_key, compute_knn)

    # ------------------------------------------------------------------
    # Arithmetic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Divide *numerator* by *denominator*, returning *default* when
        *denominator* is effectively zero (< 1e-10)."""
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator
