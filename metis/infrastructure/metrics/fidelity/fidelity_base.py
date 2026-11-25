"""
Base classes for fidelity metrics.

This module provides the unified hierarchy of base classes for all fidelity metrics:

    BaseFidelityMetric (abstract)
        │
        ├── ColumnFidelityMetric (per-column metrics)
        │       │
        │       ├── NumericColumnMetric (numeric columns only) → requires_data="num"
        │       │       └── TailMetric (alias for backward compat)
        │       │
        │       ├── CategoricalColumnMetric (categorical columns only) → requires_data="cat"
        │       │
        │       └── UniversalColumnMetric (all column types) → requires_data="both"
        │
        └── PairFidelityMetric (column pair metrics - conditional)
                │
                ├── NumNumPairMetric (numeric-numeric) → requires_data="num"
                ├── NumCatPairMetric (numeric-categorical) → requires_data="both"
                └── CatCatPairMetric (categorical-categorical) → requires_data="cat"

All metrics follow the fit/compute pattern and produce normalized scores in [0, 1].

The `requires_data` attribute indicates which transformed DataFrame(s) the metric needs:
    - "num": Only numeric DataFrame (df_num)
    - "cat": Only categorical DataFrame (df_cat)
    - "both": Both DataFrames (will receive df_full = concat of cat + num)
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from metis.domain.entities import MetricResult
from metis.shared.normalization import normalize_metric_value
from metis.shared.results import ColumnMetricResult

# type for requires_data attribute
DataRequirement = Literal["cat", "num", "both"]

# Suffix the caster gives to the numeric shadow of an ordinal column.
ORDINAL_SHADOW_SUFFIX = "__ord"


def standardise_with_missingness(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Joint-distribution view of two numeric tables that may have gaps.

    Both tables are z-scored with the real (``X``) mean and standard
    deviation of the observed values; a gap becomes 0 (the real mean) and
    every column with a gap in either table gets a 0/1 missingness indicator,
    so the rows keep their observed values and the pattern of gaps is part of
    the joint distribution (the same encoding as the privacy metrics'
    ``MixedRecordEncoder``).  The rows with any gap used to be dropped, which
    left a small, biased subset when gaps were spread over many columns.
    Without gaps the result is the former standardisation exactly.

    Returns ``(X_std, Y_std, n_indicators)``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    mx, my = np.isnan(X), np.isnan(Y)
    with warnings.catch_warnings():  # an all-missing real column: mean/std of nothing
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 1.0) + 1e-10
    Xs = np.where(mx, 0.0, (X - mean) / std)
    Ys = np.where(my, 0.0, (Y - mean) / std)
    gaps = mx.any(axis=0) | my.any(axis=0)
    if gaps.any():
        Xs = np.hstack([Xs, mx[:, gaps].astype(float)])
        Ys = np.hstack([Ys, my[:, gaps].astype(float)])
    return Xs, Ys, int(gaps.sum())


def relative_difference(diff: float, *scales: float) -> float:
    """``|diff|`` in units of the first positive reference scale of the real column.

    ``scales`` are tried in order (e.g. IQR, then standard deviation): a
    zero-inflated column has IQR 0 but a positive standard deviation. When
    every scale is 0 the real column is constant and any difference is total:
    0 if there is none, ``inf`` otherwise. The value never depends on the unit
    of the column (dividing by nothing used to return the raw difference, so
    the score of a constant or zero-inflated column changed with its unit).
    """
    diff = abs(float(diff))
    for raw_scale in scales:
        scale = float(raw_scale)
        if np.isfinite(scale) and scale > 0:
            return diff / scale
    return 0.0 if diff == 0 else float("inf")


class BaseFidelityMetric(ABC):
    """
    Abstract base class for all fidelity metrics.

    Provides common infrastructure for comparing real and synthetic datasets.

    Attributes:
        name: Short identifier for the metric
        is_distance: If True, lower raw values indicate better quality
        requires_data: Which transformed data this metric requires ("cat", "num", or "both")
    """

    name: str = "base_fidelity"
    is_distance: bool = True
    requires_data: DataRequirement = "both"  # Default to needing both

    # Instance attributes (defined here for type checking)
    _real_data: pd.DataFrame | None
    _synth_data: pd.DataFrame | None
    _context: dict

    def __init__(self):
        self._real_data = None
        self._synth_data = None
        self._context = {}

    def fit(
        self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: dict | None = None
    ) -> BaseFidelityMetric:
        """
        Initialize with data.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            context: Optional execution context (contains seed, etc.)

        Returns:
            Self for method chaining
        """
        self._real_data = real_data
        self._synth_data = synth_data
        self._context = context or {}
        self._on_fit()
        return self

    def _on_fit(self) -> None:
        """Hook for subclasses to perform additional setup after fit."""
        return

    @property
    def metric_id(self) -> str:
        """Fully-qualified id used by the central normalization registry."""
        return f"fidelity.{self.name}"

    @property
    def seed(self) -> int:
        """Seed from the execution context (42 when none was given)."""
        seed = self._context.get("seed")
        return 42 if seed is None else int(seed)

    def _cached(self, frame: pd.DataFrame, key: str, compute_fn: Callable[[], Any]) -> Any:
        """Memoise ``compute_fn`` in the shared ``StatsStore`` of the context.

        The key is tied to the identity of ``frame`` (the same transformed slice is
        handed to every metric of a run), and the frame itself is stored next to
        the value: a hit is only accepted when it refers to the very same object,
        so a store that outlives its data can never return a stale value.
        Without a store the value is simply computed.
        """
        store = self._context.get("stats_store")
        if store is None:
            return compute_fn()
        full_key = f"fidelity|{id(frame)}|{key}"
        cached_frame, value = store.get_or_compute(full_key, lambda: (frame, compute_fn()))
        if cached_frame is not frame:
            return compute_fn()
        return value

    @staticmethod
    def _ordinal_shadows(columns: set[str]) -> set[str]:
        """Numeric ``<col>__ord`` shadows whose categorical twin is also present.

        Ordinal columns reach the ``full`` slice twice (categorical and
        ``__ord``). When both are present the shadow is dropped so the column is
        neither counted twice nor paired with itself.
        """
        n = len(ORDINAL_SHADOW_SUFFIX)
        return {
            c
            for c in columns
            if isinstance(c, str) and c.endswith(ORDINAL_SHADOW_SUFFIX) and c[:-n] in columns
        }

    def _get_numeric_columns(self, include_ordinal_ranks: bool = False) -> list[str]:
        """Get common numeric columns between real and synthetic data.

        The rank view ``<col>__ord`` of an ordinal is left out when its
        categorical twin is present (the metric then sees the ordinal through
        the twin) unless *include_ordinal_ranks*: metrics that only look at
        numeric columns and measure distances or order (MMD, energy, outliers)
        must see the ordinal through its rank, or they ignore it altogether.
        """
        if self._real_data is None or self._synth_data is None:
            return []
        real_num = set(self._real_data.select_dtypes(include=[np.number]).columns)
        synth_num = set(self._synth_data.select_dtypes(include=[np.number]).columns)
        common = set(self._real_data.columns) & set(self._synth_data.columns)
        if include_ordinal_ranks:
            return sorted(real_num & synth_num)
        return sorted((real_num & synth_num) - self._ordinal_shadows(common))

    def _get_categorical_columns(self) -> list[str]:
        """Get common categorical columns between real and synthetic data."""
        if self._real_data is None or self._synth_data is None:
            return []
        real_cat = set(self._real_data.select_dtypes(include=["object", "category"]).columns)
        synth_cat = set(self._synth_data.select_dtypes(include=["object", "category"]).columns)
        return sorted(real_cat & synth_cat)

    def _get_all_columns(self) -> list[str]:
        """Get all common columns between real and synthetic data."""
        if self._real_data is None or self._synth_data is None:
            return []
        common = set(self._real_data.columns) & set(self._synth_data.columns)
        return sorted(common - self._ordinal_shadows(common))

    def _pair_categorical_columns(self) -> list[str]:
        """Categorical columns for pair metrics."""
        return self._get_categorical_columns()


# =============================================================================
# Column-level metrics (marginal)
# =============================================================================


class ColumnFidelityMetric(BaseFidelityMetric):
    """
    Base class for per-column fidelity metrics.

    Computes a metric for each applicable column independently and normalizes
    each one on an absolute scale with the central registry
    (``normalize_metric_value("fidelity.<name>", value)``). Scale-dependent raw
    values are first made scale-free by ``_scale_free_value`` (e.g. a
    Wasserstein distance is divided by the real column's standard deviation),
    so a column's score never depends on how the other columns did.

    Subclasses must implement:
        - _compute_column(): Compute raw metric for a single column
        - _get_applicable_columns(): Define which columns this metric applies to
    """

    def __init__(self):
        super().__init__()
        self._columns: list[str] = []
        self._results: dict[str, ColumnMetricResult] = {}

    def _on_fit(self) -> None:
        """Identify applicable columns on fit."""
        self._columns = self._get_applicable_columns()
        self._results = {}

    @abstractmethod
    def _get_applicable_columns(self) -> list[str]:
        """
        Get columns applicable to this metric.

        Returns:
            list of column names this metric should be computed for
        """
        pass

    def _scale_free_value(self, raw_value: float, real_col: pd.Series) -> float:  # noqa: ARG002
        """Express ``raw_value`` in units that do not depend on the column scale.

        Default: the raw value is already scale-free (KS, KL on bins, ...).
        Metrics measured in the units of the column override this.
        """
        return raw_value

    @abstractmethod
    def _compute_column(self, real_col: pd.Series, synth_col: pd.Series) -> float:
        """
        Compute the raw metric value for a single column.

        Args:
            real_col: Column from real dataset (NaN values already removed)
            synth_col: Column from synthetic dataset (NaN values already removed)

        Returns:
            Raw metric value (interpretation depends on is_distance)
        """
        pass

    def compute_all_columns(self) -> dict[str, ColumnMetricResult]:
        """
        Compute metric for all applicable columns.

        Returns:
            Dictionary mapping column names to their metric results
        """
        if self._real_data is None or self._synth_data is None:
            raise ValueError("Must call fit() before compute_all_columns()")

        metric_id = self.metric_id
        self._scale_free_values: dict[str, float] = {}
        # Columns the real table barely observes (all or all-but-one missing):
        # there is no real marginal to compare with, the column does not apply.
        self._unobserved_columns: list[str] = []

        for col in self._columns:
            try:
                # Each column with its observed values.
                real_col = self._real_data[col].dropna()
                synth_col = self._synth_data[col].dropna()

                if len(real_col) < 2:
                    self._unobserved_columns.append(col)
                    self._results[col] = ColumnMetricResult.invalid(
                        col, "Real column has fewer than 2 observed values (not applicable)"
                    )
                    continue
                if len(synth_col) < 2:
                    self._results[col] = ColumnMetricResult.invalid(col, "Insufficient data points")
                    continue

                raw_value = float(self._compute_column(real_col, synth_col))
                if np.isnan(raw_value):
                    self._results[col] = ColumnMetricResult.invalid(col, "Raw value is NaN")
                    continue
                scale_free = float(self._scale_free_value(raw_value, real_col))
                normalized = float(normalize_metric_value(metric_id, scale_free))
                if np.isnan(normalized):
                    self._results[col] = ColumnMetricResult.invalid(
                        col, f"Normalization returned NaN (raw={raw_value})"
                    )
                    continue
                self._scale_free_values[col] = scale_free
                self._results[col] = ColumnMetricResult(
                    column=col,
                    raw_value=raw_value,
                    normalized_value=normalized,
                    is_valid=True,
                )

            except Exception as e:
                self._results[col] = ColumnMetricResult.invalid(col, str(e))

        return self._results

    def compute(self) -> MetricResult:
        """
        Compute the metric for all applicable columns and return a MetricResult.

        The value is the mean normalized score of the valid columns. When no
        column could be scored the value is NaN and ``details["error"]`` says
        why, so the aggregator excludes it instead of reading a silent 0.
        """
        results = self.compute_all_columns()
        normalized_scores = [r.normalized_value for r in results.values() if r.is_valid]

        details: dict[str, Any] = {}
        for col, res in results.items():
            entry = res.to_dict()
            if col in self._scale_free_values:
                entry["scale_free_value"] = self._scale_free_values[col]
            details[col] = entry

        if not self._columns:
            return MetricResult.not_applicable(
                self.metric_id, "fidelity", f"{self.metric_id}: no column of the type it measures"
            )
        if not normalized_scores and len(self._unobserved_columns) == len(self._columns):
            return MetricResult.not_applicable(
                self.metric_id,
                "fidelity",
                f"{self.metric_id}: no column observed in the real data "
                f"({len(self._columns)} candidate(s), all missing)",
            )
        if normalized_scores:
            value = float(np.mean(normalized_scores))
        else:
            value = float("nan")
            details["error"] = (
                f"No valid columns for {self.metric_id} ({len(self._columns)} applicable, 0 scored)"
            )

        return MetricResult(
            id=self.metric_id,
            value=value,
            details=details,
            family="fidelity",
            purpose_tags=set(),
        )

    def get_normalized_scores(self) -> dict[str, float]:
        """
        Get normalized scores for all columns.

        Returns:
            Dictionary mapping column names to normalized scores in [0, 1]
        """
        return {
            col: result.normalized_value for col, result in self._results.items() if result.is_valid
        }

    def get_raw_values(self) -> dict[str, float]:
        """
        Get raw metric values for all columns.

        Returns:
            Dictionary mapping column names to raw metric values
        """
        return {col: result.raw_value for col, result in self._results.items() if result.is_valid}


class NumericColumnMetric(ColumnFidelityMetric):
    """
    Base class for metrics that work only on numeric columns.

    Used for:
        - Scale metrics (mean, median, MAD, IQR, Cohen's d)
        - Tail metrics (KS, Wasserstein, Hellinger, etc.)

    Subclasses only need to implement:
        - _compute_column(): Compute raw metric for a single column
        - name: str attribute with the metric name
        - is_distance: bool attribute (True if lower is better)
    """

    requires_data: DataRequirement = "num"

    def _get_applicable_columns(self) -> list[str]:
        """Get numeric columns only."""
        return self._get_numeric_columns()


class CategoricalColumnMetric(ColumnFidelityMetric):
    """
    Base class for metrics that work only on categorical columns.

    Used for categorical-specific divergence metrics.
    """

    requires_data: DataRequirement = "cat"

    def _get_applicable_columns(self) -> list[str]:
        """Get categorical columns only."""
        return self._get_categorical_columns()


class UniversalColumnMetric(ColumnFidelityMetric):
    """
    Base class for metrics that work on all column types.

    Used for:
        - Coverage metrics (TVD, KL, JS, PSI, Entropy)
        - Any metric that can handle both numeric and categorical

    Subclasses only need to implement:
        - _compute_column(): Compute raw metric for a single column
        - name: str attribute with the metric name
        - is_distance: bool attribute (True if lower is better)
    """

    requires_data: DataRequirement = "both"

    def _get_applicable_columns(self) -> list[str]:
        """Get all common columns."""
        return self._get_all_columns()


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# These aliases maintain backward compatibility with existing code
TailMetric = NumericColumnMetric
MarginalMetric = NumericColumnMetric
CategoricalMarginalMetric = CategoricalColumnMetric
UniversalMarginalMetric = UniversalColumnMetric

# =============================================================================
# Pair-level metrics (conditional)
# =============================================================================


class PairFidelityMetric(BaseFidelityMetric):
    """
    Base class for column pair fidelity metrics (conditional/bivariate).

    Computes metrics for pairs of columns to measure relationship preservation.

    Subclasses must implement:
        - _compute_pair(): Compute metric for a column pair
        - _get_applicable_pairs(): Define which pairs this metric applies to

    Pair delta normalization
    ------------------------
    Each pair contributes a delta ``δ = |real_val - synth_val|`` to the
    aggregated score. Two unification policies are available:

    * ``pair_delta_normalization = "auto"`` (default): the metric's central
      ``METRIC_NORMALIZATION_MAP`` entry is used via
      :func:`metis.shared.normalization.normalize_metric_value`. This keeps
      pair-metric scoring consistent with how the same metric is normalized
      everywhere else (for example ``fidelity.chi2_stat`` falls under
      ``UNBOUNDED_DISTANCE`` rather than the historical saturating
      ``1 - min(δ, 1)`` rule).
    * ``pair_delta_normalization = "linear_clip"`` (legacy): retained for
      metrics whose delta really is in ``[0, 1]`` (e.g. correlation deltas)
      and whose contributors expect ``1 - min(δ, 1)`` semantics.

    Subclasses may override the class attribute below if they need a
    different policy without touching the central registry.
    """

    pair_delta_normalization: str = "auto"

    def __init__(self):
        super().__init__()
        self._pairs: list[tuple[str, str]] = []

    def _on_fit(self) -> None:
        """Identify applicable pairs on fit."""
        self._pairs = self._get_applicable_pairs()

    def _prepare_pairs(self) -> None:
        """Hook run once before the pair loop to precompute shared statistics.

        Subclasses use it to compute everything that does not depend on the
        pair (a correlation matrix, per-column centred distance matrices...).
        """
        return

    @staticmethod
    def _pair_view(
        frame: pd.DataFrame, c1: str, c2: str, complete: dict[str, bool]
    ) -> tuple[pd.Series, pd.Series]:
        """Both columns restricted to the rows where the two are observed.

        When neither column has missing values the columns are returned as-is
        (no ``dropna``/``intersection``/``.loc`` per pair).
        """
        col1, col2 = frame[c1], frame[c2]
        if complete[c1] and complete[c2]:
            return col1, col2
        mask = col1.notna().to_numpy() & col2.notna().to_numpy()
        return col1[mask], col2[mask]

    def compute(self) -> MetricResult:
        """
        Compute the metric for all applicable pairs and return a MetricResult.

        Returns:
            MetricResult with the mean normalized score as value.
        """
        if self._real_data is None or self._synth_data is None:
            raise ValueError("Must call fit() before compute()")

        normalized_scores = []
        details: dict[str, Any] = {}
        skipped_count = 0
        error_count = 0

        # Per-column facts computed once instead of once per pair.
        cols = sorted({c for pair in self._pairs for c in pair})
        real_complete = self._real_data[cols].notna().all().to_dict() if cols else {}
        synth_complete = self._synth_data[cols].notna().all().to_dict() if cols else {}
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self._real_data[c])]
        num_set = set(num_cols)
        real_std = self._real_data[num_cols].std().to_dict() if num_cols else {}
        synth_std = self._synth_data[num_cols].std().to_dict() if num_cols else {}

        def _std(col: pd.Series, name: str, full: bool, cache: dict[str, float]) -> float:
            return cache[name] if full else col.std()

        self._prepare_pairs()

        for c1, c2 in self._pairs:
            pair_key = f"{c1}|{c2}"
            try:
                real_col1, real_col2 = self._pair_view(self._real_data, c1, c2, real_complete)
                synth_col1, synth_col2 = self._pair_view(self._synth_data, c1, c2, synth_complete)
                both_real = real_complete[c1] and real_complete[c2]
                both_synth = synth_complete[c1] and synth_complete[c2]

                if len(real_col1) < 10 or len(synth_col1) < 10:
                    skipped_count += 1
                    details[pair_key] = {
                        "skipped": True,
                        "reason": f"Insufficient data (real={len(real_col1)}, synth={len(synth_col1)})",
                    }
                    continue

                # Check for constant columns (zero variance causes NaN correlations)
                # Only check for numeric columns - categorical columns don't have std()
                is_numeric_c1 = c1 in num_set
                is_numeric_c2 = c2 in num_set

                if is_numeric_c1 and is_numeric_c2:
                    real_c1_std = _std(real_col1, c1, both_real, real_std)
                    real_c2_std = _std(real_col2, c2, both_real, real_std)
                    synth_c1_std = _std(synth_col1, c1, both_synth, synth_std)
                    synth_c2_std = _std(synth_col2, c2, both_synth, synth_std)

                    if real_c1_std == 0 or real_c2_std == 0:
                        skipped_count += 1
                        details[pair_key] = {
                            "skipped": True,
                            "reason": f"Constant column in real data ({c1} std={real_c1_std}, {c2} std={real_c2_std})",
                        }
                        continue

                    if synth_c1_std == 0 or synth_c2_std == 0:
                        skipped_count += 1
                        details[pair_key] = {
                            "skipped": True,
                            "reason": f"Constant column in synth data ({c1} std={synth_c1_std}, {c2} std={synth_c2_std})",
                        }
                        continue

                elif is_numeric_c1 or is_numeric_c2:
                    # For num-cat pairs, only check the numeric column
                    num_name = c1 if is_numeric_c1 else c2
                    real_std_v = _std(
                        real_col1 if is_numeric_c1 else real_col2, num_name, both_real, real_std
                    )
                    synth_std_v = _std(
                        synth_col1 if is_numeric_c1 else synth_col2, num_name, both_synth, synth_std
                    )

                    if real_std_v == 0:
                        skipped_count += 1
                        details[pair_key] = {
                            "skipped": True,
                            "reason": f"Constant numeric column in real data ({num_name} std=0)",
                        }
                        continue

                    if synth_std_v == 0:
                        skipped_count += 1
                        details[pair_key] = {
                            "skipped": True,
                            "reason": f"Constant numeric column in synth data ({num_name} std=0)",
                        }
                        continue

                # For cat-cat pairs, we don't skip based on unique values
                # The metric calculation will handle the case where there's no variation
                # (e.g., Cramer's V will be 0 if one column has only 1 category)

                real_val, synth_val = self._compute_pair(
                    real_col1,
                    real_col2,
                    synth_col1,
                    synth_col2,
                )

                # Check for NaN results
                if np.isnan(real_val) or np.isnan(synth_val):
                    error_count += 1
                    details[pair_key] = {
                        "error": f"NaN result (real={real_val}, synth={synth_val})",
                    }
                    continue

                # Normalize delta using the unified policy declared by the
                # subclass. See ``PairFidelityMetric.pair_delta_normalization``.
                delta = abs(real_val - synth_val)
                normalized = self._normalize_pair_delta(delta)
                normalized_scores.append(normalized)

                details[pair_key] = {
                    "real_value": real_val,
                    "synth_value": synth_val,
                    "delta": delta,
                    "normalized_value": normalized,
                }

            except Exception as e:
                error_count += 1
                details[pair_key] = {"error": str(e)}

        if not normalized_scores and error_count == 0:
            # No pair of the right types, or every pair skipped as outside the
            # metric's domain (e.g. point-biserial needs a binary column).
            return MetricResult.not_applicable(
                self.metric_id,
                "fidelity",
                f"{self.metric_id}: no column pair it applies to "
                f"({len(self._pairs)} candidate(s), {skipped_count} skipped)",
                _summary={"total_pairs": len(self._pairs), "skipped_pairs": skipped_count},
            )
        # Compute aggregate value (only from valid scores)
        if normalized_scores:
            value = float(np.mean(normalized_scores))
        else:
            value = float("nan")
            details["error"] = (
                f"No valid pairs for {self.metric_id} ({len(self._pairs)} applicable, "
                f"{skipped_count} skipped, {error_count} errored)"
            )

        # Add summary statistics
        total_pairs = len(self._pairs)
        valid_pairs = len(normalized_scores)
        details["_summary"] = {
            "total_pairs": total_pairs,
            "valid_pairs": valid_pairs,
            "skipped_pairs": skipped_count,
            "error_pairs": error_count,
            "coverage_pct": (round(100 * valid_pairs / total_pairs, 1) if total_pairs > 0 else 0.0),
        }

        return MetricResult(
            id=self.metric_id,
            value=value,
            details=details,
            family="fidelity",
            purpose_tags=set(),
        )

    @abstractmethod
    def _get_applicable_pairs(self) -> list[tuple[str, str]]:
        """
        Get column pairs applicable to this metric.

        Returns:
            list of (col1, col2) tuples
        """
        pass

    @abstractmethod
    def _compute_pair(
        self,
        real_col1: pd.Series,
        real_col2: pd.Series,
        synth_col1: pd.Series,
        synth_col2: pd.Series,
    ) -> tuple[float, float]:
        """
        Compute the metric for a column pair.

        Args:
            real_col1: First column from real dataset
            real_col2: Second column from real dataset
            synth_col1: First column from synthetic dataset
            synth_col2: Second column from synthetic dataset

        Returns:
            tuple of (real_value, synth_value)
        """
        pass

    def _normalize_pair_delta(self, delta: float) -> float:
        """Normalize a pair delta into ``[0, 1]`` (1 = best).

        Routes through :func:`metis.shared.normalization.normalize_metric_value`
        when ``pair_delta_normalization == "auto"``, so that pair scoring stays
        consistent with the central per-metric normalization registry. Falls
        back to the legacy ``1 - min(δ, 1)`` rule when the subclass opts into
        ``"linear_clip"``.
        """
        if self.pair_delta_normalization == "linear_clip":
            return 1.0 - min(float(delta), 1.0)
        return float(normalize_metric_value(self.metric_id, float(delta)))


class NumNumPairMetric(PairFidelityMetric):
    """
    Base class for numeric-numeric pair metrics.

    Used for correlation metrics (Pearson, Spearman, dCor, MI).
    """

    requires_data: DataRequirement = "num"

    def _get_applicable_pairs(self) -> list[tuple[str, str]]:
        """Get all numeric-numeric pairs."""
        numeric_cols = self._get_numeric_columns()
        return [(c1, c2) for i, c1 in enumerate(numeric_cols) for c2 in numeric_cols[i + 1 :]]


class NumCatPairMetric(PairFidelityMetric):
    """
    Base class for numeric-categorical pair metrics.

    Used for association metrics (point-biserial, eta-squared, Kruskal).
    """

    requires_data: DataRequirement = "both"

    def _get_applicable_pairs(self) -> list[tuple[str, str]]:
        """Numeric-categorical pairs, an ordinal counted once per pair of variables.

        An ordinal is paired with the numeric columns through its level
        (groups of the numeric column by level), and with the nominal
        columns through its rank ``<col>__ord`` (the rank grouped by the
        nominal categories, which keeps its order; as two categoricals that
        pair is already scored by the cat-cat metrics).  Pairs of two
        ordinals are scored by the num-num metrics on their ranks.
        """
        numeric_cols = self._get_numeric_columns()
        categorical_cols = self._pair_categorical_columns()
        pairs = [(n, c) for n in numeric_cols for c in categorical_cols]
        common = set(self._real_data.columns) & set(self._synth_data.columns)
        ranks = sorted(self._ordinal_shadows(common))
        twins = {r[: -len(ORDINAL_SHADOW_SUFFIX)] for r in ranks}
        nominal = [c for c in categorical_cols if c not in twins]
        pairs += [(r, c) for r in ranks for c in nominal]
        return pairs

    @staticmethod
    def _groups(num_col: pd.Series, cat_col: pd.Series, min_size: int = 2) -> list[np.ndarray]:
        """Values of ``num_col`` split by the categories of ``cat_col``.

        Same groups as ``DataFrame.groupby(cat)`` (groups smaller than
        ``min_size`` dropped) built with ``factorize`` + one ``argsort`` instead
        of a DataFrame and a groupby per pair.
        """
        codes, uniques = pd.factorize(cat_col, sort=True)
        values = num_col.to_numpy(dtype=float)
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes, minlength=len(uniques))
        groups = np.split(values[order], np.cumsum(counts)[:-1])
        return [g for g in groups if len(g) >= min_size]


class CatCatPairMetric(PairFidelityMetric):
    """
    Base class for categorical-categorical pair metrics.

    Used for contingency metrics (Cramér's V, Theil's U, Chi-squared).
    """

    requires_data: DataRequirement = "cat"

    def _get_applicable_pairs(self) -> list[tuple[str, str]]:
        """Get all categorical-categorical pairs."""
        categorical_cols = self._pair_categorical_columns()
        return [
            (c1, c2) for i, c1 in enumerate(categorical_cols) for c2 in categorical_cols[i + 1 :]
        ]

    def _contingency(self, side: str, col1: pd.Series, col2: pd.Series) -> np.ndarray:
        """Contingency table of two categorical columns (same as ``pd.crosstab``).

        Built with ``factorize`` + ``bincount`` and shared through the
        ``StatsStore``: Cramér's V, Theil's U and chi² ask for the very same
        tables. Only complete columns (no rows dropped for the pair) are cached.
        """
        frame = self._real_data if side == "real" else self._synth_data

        def build() -> np.ndarray:
            codes1, uniques1 = pd.factorize(col1, sort=True)
            codes2, uniques2 = pd.factorize(col2, sort=True)
            k1, k2 = len(uniques1), len(uniques2)
            counts = np.bincount(codes1 * k2 + codes2, minlength=k1 * k2)
            return counts.reshape(k1, k2)

        if frame is None or len(col1) != len(frame):
            return build()
        return self._cached(frame, f"contingency|{col1.name}|{col2.name}", build)


# =============================================================================
# Global-level metrics
# =============================================================================


class GlobalMetricNotApplicable(ValueError):
    """Raised by ``_compute_global`` when the tables have nothing the metric can measure."""


class GlobalFidelityMetric(BaseFidelityMetric):
    """
    Base class for global fidelity metrics.

    Global metrics compare entire datasets rather than individual columns or pairs.

    Subclasses must implement:
        - _compute_global(): Compute the global metric value
    """

    requires_data: DataRequirement = "both"

    def compute(self) -> MetricResult:
        """
        Compute the global metric and return a MetricResult.

        Returns:
            MetricResult with the normalized score as value.
        """
        if self._real_data is None or self._synth_data is None:
            raise ValueError("Must call fit() before compute()")

        try:
            raw_value, normalized_value, details = self._compute_global()

            return MetricResult(
                id=self.metric_id,
                value=normalized_value,
                details=details,
                family="fidelity",
                purpose_tags=set(),
            )

        except GlobalMetricNotApplicable as e:
            # Nothing to measure (e.g. no numeric column): not a failure.
            return MetricResult.not_applicable(self.metric_id, "fidelity", str(e), set())
        except Exception as e:
            return MetricResult(
                id=self.metric_id,
                value=float("nan"),
                details={"error": str(e)},
                family="fidelity",
                purpose_tags=set(),
            )

    @abstractmethod
    def _compute_global(self) -> tuple[float, float, dict]:
        """
        Compute the global metric.

        Returns:
            tuple of (raw_value, normalized_value, details_dict)
        """
        pass
