"""Step 3 — Validate preprocessed data integrity.

Single responsibility: assert data quality after preprocessing.
Raises SchemaError on hard failures, returns warnings for soft issues.
"""

import pandas as pd

from ...domain.entities import DatasetSpec, EvalPlan, TransformedData
from ...domain.errors import SchemaError
from ...infrastructure.runtime.logging import get_logger


class DataValidator:
    """Validates that preprocessed data is fit for metric evaluation."""

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    def validate(
        self,
        real: TransformedData,
        synth: TransformedData,
        spec: DatasetSpec,
        plan: EvalPlan,
    ) -> list[str]:
        """Run all validation checks.

        Returns:
            list of non-fatal warnings.

        Raises:
            SchemaError: On any hard validation failure.
        """
        warnings: list[str] = []

        self._check_not_empty(real.full, "real")
        self._check_not_empty(synth.full, "synth")
        self._check_no_nans(real.full, "real")
        self._check_no_nans(synth.full, "synth")
        self._check_min_rows(real.full, "real")
        self._check_min_rows(synth.full, "synth")
        self._check_schema_compat(real.full, synth.full)
        self._check_metric_ids(plan)

        if spec.target:
            self._check_target(real.full, synth.full, spec)

        # Utility metrics require a target column
        utility_ids = [m for m in plan.metric_ids if m.startswith("utility.")]
        if utility_ids and not spec.target:
            raise SchemaError("Target column must be specified for utility metrics")

        return warnings

    # ----- individual checks -------------------------------------------------

    @staticmethod
    def _check_not_empty(df: pd.DataFrame, label: str) -> None:
        if df.empty or len(df.columns) == 0:
            raise SchemaError(f"{label.capitalize()} data cannot be empty")

    @staticmethod
    def _check_no_nans(df: pd.DataFrame, label: str) -> None:
        nan_cols = [c for c in df.columns if df[c].isna().any()]
        if nan_cols:
            raise SchemaError(
                f"{label.capitalize()} data still contains NaN values "
                f"after preprocessing in columns: {nan_cols}"
            )

    @staticmethod
    def _check_min_rows(df: pd.DataFrame, label: str, minimum: int = 10) -> None:
        if len(df) < minimum:
            raise SchemaError(
                f"{label.capitalize()} data must have at least {minimum} rows " f"(got {len(df)})"
            )

    @staticmethod
    def _check_metric_ids(plan: EvalPlan) -> None:
        """Verify every metric_id belongs to a known family."""
        valid_prefixes = ("fidelity.", "utility.", "privacy.")
        invalid = [m for m in plan.metric_ids if not m.startswith(valid_prefixes)]
        if invalid:
            raise SchemaError(
                f"Unknown metric families in metric_ids: {invalid}. "
                f"Each metric must start with one of: {', '.join(valid_prefixes)}"
            )

    @staticmethod
    def _check_schema_compat(real: pd.DataFrame, synth: pd.DataFrame) -> None:
        real_cols = set(real.columns)
        synth_cols = set(synth.columns)
        missing = real_cols - synth_cols
        if missing:
            raise SchemaError(f"Columns missing in synthetic data: {missing}")
        extra = synth_cols - real_cols
        if extra:
            raise SchemaError(f"Extra columns in synthetic data: {extra}")

    @staticmethod
    def _check_target(real: pd.DataFrame, synth: pd.DataFrame, spec: "DatasetSpec") -> None:
        for target in spec.target_list:
            for label, df in [("real", real), ("synth", synth)]:
                if target not in df.columns:
                    raise SchemaError(
                        f"Target column '{target}' not found in {label} data",
                        column=target,
                    )
                non_null = df[target].notna().sum()
                if non_null < 10:
                    raise SchemaError(
                        f"Target '{target}' has too few non-null values in {label} data: {non_null}",
                        column=target,
                    )
