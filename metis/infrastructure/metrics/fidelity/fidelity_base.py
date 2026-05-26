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

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd

from metis.domain.entities import MetricResult
from metis.shared.normalization import (
    METRIC_NORMALIZATION_MAP,
    NormalizationType,
    normalize_metric_value,
)
from metis.shared.results import ColumnMetricResult

# type for requires_data attribute
DataRequirement = Literal["cat", "num", "both"]


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
    ) -> "BaseFidelityMetric":
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

    def _get_numeric_columns(self) -> list[str]:
        """Get common numeric columns between real and synthetic data."""
        if self._real_data is None or self._synth_data is None:
            return []
        real_num = set(self._real_data.select_dtypes(include=[np.number]).columns)
        synth_num = set(self._synth_data.select_dtypes(include=[np.number]).columns)
        return sorted(real_num & synth_num)

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
        return sorted(set(self._real_data.columns) & set(self._synth_data.columns))


# =============================================================================
# Column-level metrics (marginal)
# =============================================================================


class ColumnFidelityMetric(BaseFidelityMetric):
    """
    Base class for per-column fidelity metrics.

    Computes a metric for each applicable column independently,
    then normalizes across all columns using robust percentiles.

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

        raw_values = {}

        # First pass: compute raw values for all columns
        for col in self._columns:
            try:
                real_col = self._real_data[col].dropna()
                synth_col = self._synth_data[col].dropna()

                if len(real_col) < 2 or len(synth_col) < 2:
                    self._results[col] = ColumnMetricResult.invalid(col, "Insufficient data points")
                    continue

                raw_value = self._compute_column(real_col, synth_col)
                raw_values[col] = raw_value

            except Exception as e:
                self._results[col] = ColumnMetricResult.invalid(col, str(e))

        # Second pass: normalize using metric-specific normalization
        if raw_values:
            # Use the full metric ID for proper normalization lookup
            metric_id = f"fidelity.{self.name}"
            # Use the is_distance attribute to flip values if needed
            normalized = self._normalize_column_values(raw_values, metric_id)

            for col, raw_value in raw_values.items():
                self._results[col] = ColumnMetricResult(
                    column=col,
                    raw_value=raw_value,
                    normalized_value=normalized[col],
                    is_valid=True,
                )

        return self._results

    def _normalize_column_values(
        self, raw_values: dict[str, float], metric_id: str
    ) -> dict[str, float]:
        """
        Normalize column values using metric-specific normalization.

        For bounded metrics (ks, hellinger, tvd, js): 1 - value
        For unbounded metrics (wasserstein, kl, etc): robust normalization

        Args:
            raw_values: Dictionary of column names to raw values
            metric_id: Full metric ID (e.g., "fidelity.ks")

        Returns:
            Dictionary of normalized values in [0, 1]
        """

        if not raw_values:
            return {}

        # Check if this metric uses UNBOUNDED_DISTANCE normalization
        norm_type = METRIC_NORMALIZATION_MAP.get(metric_id)

        if norm_type == NormalizationType.UNBOUNDED_DISTANCE:
            # For unbounded metrics, use robust percentile-based normalization
            # to handle arbitrary scales
            return self._normalize_unbounded_robust(raw_values)
        # For bounded/similarity/delta metrics, use direct normalization
        return {col: normalize_metric_value(metric_id, value) for col, value in raw_values.items()}

    def _normalize_unbounded_robust(self, raw_values: dict[str, float]) -> dict[str, float]:
        """
        Robust normalization for unbounded distance metrics.

        Uses percentile-based scaling to handle arbitrary value ranges.
        Lower values = better for distance metrics.

        Args:
            raw_values: Dictionary of column names to raw values

        Returns:
            Dictionary of normalized values in [0, 1] where 1 = best
        """
        if not raw_values:
            return {}

        all_values = np.array(list(raw_values.values()))
        valid_values = all_values[~np.isnan(all_values)]

        if len(valid_values) == 0:
            return dict.fromkeys(raw_values, 0.0)

        # Use robust percentiles to determine scale
        p_min = np.percentile(valid_values, 5)
        p_max = np.percentile(valid_values, 95)

        # Avoid division by zero
        if p_max - p_min < 1e-10:
            # All values are essentially the same
            # For distance metrics: if all values are near zero, it's perfect (1.0)
            # Otherwise, it's a neutral score (0.5)
            if p_max < 1e-6:  # All values are near zero = perfect match
                return dict.fromkeys(raw_values, 1.0)
            # All values are the same but non-zero = neutral
            return dict.fromkeys(raw_values, 0.5)

        normalized = {}
        for col, raw_value in raw_values.items():
            if np.isnan(raw_value):
                normalized[col] = 0.0
                continue

            # Scale to [0, 1] using robust percentiles
            norm_val = (raw_value - p_min) / (p_max - p_min)
            norm_val = np.clip(norm_val, 0.0, 1.0)

            # For distance metrics, invert (lower is better → higher score)
            norm_val = 1.0 - norm_val

            normalized[col] = float(norm_val)

        return normalized

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

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context=None,  # noqa: ARG002 - context ignored for interface compatibility
    ) -> "NumericColumnMetric":
        """
        Initialize with data.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            context: Ignored, for interface compatibility

        Returns:
            Self for method chaining
        """
        return super().fit(real_data, synth_data)

    def compute(self) -> "MetricResult":
        """
        Compute the metric for all applicable columns and return a MetricResult.

        Returns:
            MetricResult with the mean normalized score as value, and details per column.
        """
        results = self.compute_all_columns()

        # Get normalized scores for valid columns
        normalized_scores = [r.normalized_value for r in results.values() if r.is_valid]

        # Aggregate: mean normalized score, or 0.0 if none valid
        value = float(np.mean(normalized_scores)) if normalized_scores else 0.0

        # Prepare details: dict of column -> result dict
        details = {col: res.to_dict() for col, res in results.items()}

        # Return MetricResult as expected by orchestrator
        return MetricResult(
            id=self.name,
            value=value,
            details=details,
            family="fidelity",
            purpose_tags=set(),
        )


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

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context=None,  # noqa: ARG002 - context ignored for interface compatibility
    ) -> "UniversalColumnMetric":
        """
        Initialize with data.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            context: Ignored, for interface compatibility

        Returns:
            Self for method chaining
        """
        return super().fit(real_data, synth_data)

    def compute(self) -> "MetricResult":
        """
        Compute the metric for all applicable columns and return a MetricResult.

        Returns:
            MetricResult with the mean normalized score as value, and details per column.
        """
        results = self.compute_all_columns()

        # Get normalized scores for valid columns
        normalized_scores = [r.normalized_value for r in results.values() if r.is_valid]

        # Aggregate: mean normalized score, or 0.0 if none valid
        value = float(np.mean(normalized_scores)) if normalized_scores else 0.0

        # Prepare details: dict of column -> result dict
        details = {col: res.to_dict() for col, res in results.items()}

        # Return MetricResult as expected by orchestrator
        return MetricResult(
            id=self.name,
            value=value,
            details=details,
            family="fidelity",
            purpose_tags=set(),
        )


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

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context=None,  # noqa: ARG002 - context ignored for interface compatibility
    ) -> "PairFidelityMetric":
        """
        Initialize with data.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            context: Ignored, for interface compatibility

        Returns:
            Self for method chaining
        """
        return super().fit(real_data, synth_data)

    def compute(self) -> "MetricResult":
        """
        Compute the metric for all applicable pairs and return a MetricResult.

        Returns:
            MetricResult with the mean normalized score as value.
        """
        if self._real_data is None or self._synth_data is None:
            raise ValueError("Must call fit() before compute()")

        normalized_scores = []
        details = {}
        skipped_count = 0
        error_count = 0

        for c1, c2 in self._pairs:
            pair_key = f"{c1}|{c2}"
            try:
                real_col1 = self._real_data[c1].dropna()
                real_col2 = self._real_data[c2].dropna()
                synth_col1 = self._synth_data[c1].dropna()
                synth_col2 = self._synth_data[c2].dropna()

                # Get common indices after dropna
                real_idx = real_col1.index.intersection(real_col2.index)
                synth_idx = synth_col1.index.intersection(synth_col2.index)

                if len(real_idx) < 10 or len(synth_idx) < 10:
                    skipped_count += 1
                    details[pair_key] = {
                        "skipped": True,
                        "reason": f"Insufficient data (real={len(real_idx)}, synth={len(synth_idx)})",
                    }
                    continue

                # Check for constant columns (zero variance causes NaN correlations)
                # Only check for numeric columns - categorical columns don't have std()
                is_numeric_c1 = pd.api.types.is_numeric_dtype(real_col1)
                is_numeric_c2 = pd.api.types.is_numeric_dtype(real_col2)

                if is_numeric_c1 and is_numeric_c2:
                    # For numeric-numeric pairs, check std
                    real_c1_std = real_col1.loc[real_idx].std()
                    real_c2_std = real_col2.loc[real_idx].std()
                    synth_c1_std = synth_col1.loc[synth_idx].std()
                    synth_c2_std = synth_col2.loc[synth_idx].std()

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
                    num_col_real = real_col1 if is_numeric_c1 else real_col2
                    num_col_synth = synth_col1 if is_numeric_c1 else synth_col2
                    num_name = c1 if is_numeric_c1 else c2

                    real_std = num_col_real.loc[real_idx].std()
                    synth_std = num_col_synth.loc[synth_idx].std()

                    if real_std == 0:
                        skipped_count += 1
                        details[pair_key] = {
                            "skipped": True,
                            "reason": f"Constant numeric column in real data ({num_name} std=0)",
                        }
                        continue

                    if synth_std == 0:
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
                    real_col1.loc[real_idx],
                    real_col2.loc[real_idx],
                    synth_col1.loc[synth_idx],
                    synth_col2.loc[synth_idx],
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

        # Compute aggregate value (only from valid scores)
        if normalized_scores:
            value = float(np.mean(normalized_scores))
        else:
            value = float("nan")

        # Add summary statistics
        total_pairs = len(self._pairs)
        valid_pairs = len(normalized_scores)
        details["_summary"] = {
            "total_pairs": total_pairs,
            "valid_pairs": valid_pairs,
            "skipped_pairs": skipped_count,
            "error_pairs": error_count,
            "coverage_pct": (round(100 * valid_pairs / total_pairs, 1) if total_pairs > 0 else 0.0),
            "not_applicable": valid_pairs == 0,
        }

        return MetricResult(
            id=self.name,
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
        # Lazy import to avoid a circular dependency: normalization.py is in
        # ``metis.shared`` which is imported widely at startup.
        from metis.shared.normalization import normalize_metric_value

        return float(normalize_metric_value(self.name, float(delta)))


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
        """Get all numeric-categorical pairs."""
        numeric_cols = self._get_numeric_columns()
        categorical_cols = self._get_categorical_columns()
        return [(n, c) for n in numeric_cols for c in categorical_cols]


class CatCatPairMetric(PairFidelityMetric):
    """
    Base class for categorical-categorical pair metrics.

    Used for contingency metrics (Cramér's V, Theil's U, Chi-squared).
    """

    requires_data: DataRequirement = "cat"

    def _get_applicable_pairs(self) -> list[tuple[str, str]]:
        """Get all categorical-categorical pairs."""
        categorical_cols = self._get_categorical_columns()
        return [
            (c1, c2) for i, c1 in enumerate(categorical_cols) for c2 in categorical_cols[i + 1 :]
        ]


# =============================================================================
# Global-level metrics
# =============================================================================


class GlobalFidelityMetric(BaseFidelityMetric):
    """
    Base class for global fidelity metrics.

    Global metrics compare entire datasets rather than individual columns or pairs.

    Subclasses must implement:
        - _compute_global(): Compute the global metric value
    """

    requires_data: DataRequirement = "both"

    def fit(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        context=None,  # noqa: ARG002 - context ignored for interface compatibility
    ) -> "GlobalFidelityMetric":
        """
        Initialize with data.

        Args:
            real_data: Original dataset
            synth_data: Synthetic dataset
            context: Ignored, for interface compatibility

        Returns:
            Self for method chaining
        """
        return super().fit(real_data, synth_data)

    def compute(self) -> "MetricResult":
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
                id=self.name,
                value=normalized_value,
                details=details,
                family="fidelity",
                purpose_tags=set(),
            )

        except Exception as e:
            return MetricResult(
                id=self.name,
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
