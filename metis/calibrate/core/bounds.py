"""
Calibration bounds storage and loading.

Stores empirical upper and lower bounds for family-level scores (fidelity, privacy, utility)
to normalize scores between worst-case (uniform noise) and best-case (real vs real).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CalibrationBounds:
    """
    Storage for calibrated family-level metric bounds.

    Stores upper bounds (real vs real) and lower bounds (real vs noise)
    for each metric family (fidelity, privacy, utility).
    """

    def __init__(self):
        """Initialize empty bounds storage."""
        self.bounds: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}
        self.optimal_aggregators: dict[str, Any] | None = None
        self.metric_details: dict[str, Any] | None = None

    def set_bounds(
        self,
        family: str,
        lower_bound: float,
        upper_bound: float,
        lower_iterations: list[float] | None = None,
        upper_iterations: list[float] | None = None,
        inverted: bool = False,
    ) -> None:
        """
        set bounds for a specific metric family.

        Args:
            family: Family name ("fidelity", "privacy", "utility")
            lower_bound: Lower bound (worst-case scenario)
            upper_bound: Upper bound (best-case scenario)
            lower_iterations: All lower bound iteration values
            upper_iterations: All upper bound iteration values
            inverted: If True, the aggregator produces scores with inverted
                semantics (lower raw = better).  Normalization will flip the
                result so that 1 still means "best".

        Note:
            The caller (MetricCalibrator) is responsible for ensuring that
            lower_bound <= upper_bound by swapping iteration labels when needed
            (e.g., for privacy metrics with inverted calibration semantics).
        """
        # Validate bounds are in correct order
        if upper_bound < lower_bound:
            import warnings

            warnings.warn(
                f"\n{'=' * 80}\n"
                f"ERROR: INVALID BOUNDS FOR {family.upper()}!\n"
                f"{'=' * 80}\n"
                f"Upper bound ({upper_bound:.4f}) < Lower bound ({lower_bound:.4f})\n"
                f"This should have been handled by the calibrator.\n"
                f"Bounds will be stored as-is but normalization will fail.\n"
                f"{'=' * 80}\n",
                category=RuntimeWarning,
                stacklevel=2,
            )

        self.bounds[family] = {
            "lower": float(lower_bound),
            "upper": float(upper_bound),
            "lower_iterations": [float(x) for x in lower_iterations] if lower_iterations else [],
            "upper_iterations": [float(x) for x in upper_iterations] if upper_iterations else [],
            "inverted": inverted,
        }

    def get_bounds(self, family: str) -> tuple[float, float]:
        """
        Get bounds tuple for a specific family.

        Args:
            family: Family name

        Returns:
            tuple of (lower_bound, upper_bound)

        Raises:
            KeyError: If family not found
        """
        if family not in self.bounds:
            raise KeyError(f"Family '{family}' not found in calibration bounds")

        data = self.bounds[family]
        return (data["lower"], data["upper"])

    def get_all_families(self) -> list[str]:
        """
        Get list of all families with calibrated bounds.

        Returns:
            list of family names
        """
        return list(self.bounds.keys())

    def normalize_with_bounds(self, family: str, raw_value: float) -> float:
        """
        Normalize a raw family score using calibrated bounds.

        Formula: (value - lower) / (upper - lower), clipped to [0, 1]
        When the aggregator has inverted semantics (lower raw = better),
        the result is flipped: 1 - normalized.

        Args:
            family: Family name ("fidelity", "privacy", "utility")
            raw_value: Raw family score to normalize

        Returns:
            Normalized value in [0, 1] where 1 = best
        """
        if family not in self.bounds:
            raise ValueError(
                f"No calibration bounds found for family '{family}'. "
                f"Available families: {list(self.bounds.keys())}"
            )

        bounds = self.bounds[family]
        lower = bounds["lower"]
        upper = bounds["upper"]
        inverted = bounds.get("inverted", False)

        # Handle edge case where bounds are identical
        if abs(upper - lower) < 1e-10:
            return 1.0 if raw_value >= upper else 0.0

        # Normalize and clip
        normalized = (raw_value - lower) / (upper - lower)
        normalized = float(np.clip(normalized, 0.0, 1.0))

        # When the aggregator inverts score semantics (e.g. SSD on some
        # datasets produces lower scores for better fidelity), flip so
        # that 1 still means "best quality".
        if inverted:
            normalized = 1.0 - normalized

        return normalized

    def save(self, filepath: str) -> None:
        """
        Save bounds to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp to metadata
        self.metadata["saved_at"] = datetime.now().isoformat()

        # Ensure critical metadata is present (for cache validation)
        if "data_fingerprint" not in self.metadata:
            logger.warning(
                "Saving calibration bounds without data_fingerprint. "
                "Cache validation will not work properly."
            )

        data = {
            "bounds": self.bounds,
            "metadata": self.metadata,
        }
        if self.optimal_aggregators:
            data["optimal_aggregators"] = self.optimal_aggregators
        if self.metric_details:
            data["metric_details"] = self.metric_details

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _metric_details_to_tuner_input(
        metric_details: dict[str, Any] | None,
    ) -> tuple[dict[str, list[dict[str, float]]], dict[str, list[dict[str, float]]]]:
        """Reconstruct tuner inputs from persisted metric details."""
        upper_metric_data: dict[str, list[dict[str, float]]] = {}
        lower_metric_data: dict[str, list[dict[str, float]]] = {}

        if not isinstance(metric_details, dict):
            return upper_metric_data, lower_metric_data

        def _extract_iterations(iterations: Any) -> list[dict[str, float]]:
            extracted: list[dict[str, float]] = []
            if not isinstance(iterations, list):
                return extracted

            for iteration_payload in iterations:
                if not isinstance(iteration_payload, dict):
                    continue

                values: dict[str, float] = {}
                for metric_id, payload in iteration_payload.items():
                    if metric_id == "iteration":
                        continue

                    normalized_value = (
                        payload.get("normalized") if isinstance(payload, dict) else payload
                    )
                    if normalized_value is None:
                        continue

                    try:
                        value = float(normalized_value)
                    except (TypeError, ValueError):
                        continue

                    if np.isfinite(value):
                        values[metric_id] = value

                if values:
                    extracted.append(values)

            return extracted

        for family, family_details in metric_details.items():
            if not isinstance(family_details, dict):
                continue

            upper_iterations = _extract_iterations(family_details.get("upper_iterations"))
            lower_iterations = _extract_iterations(family_details.get("lower_iterations"))

            if upper_iterations and lower_iterations:
                upper_metric_data[family] = upper_iterations
                lower_metric_data[family] = lower_iterations

        return upper_metric_data, lower_metric_data

    def _refresh_optimal_aggregators_from_metric_details(self) -> None:
        """Refresh stale cached aggregators when persisted L4 is degenerate."""
        if not isinstance(self.optimal_aggregators, dict) or not self.metric_details:
            return

        composite_name = self.optimal_aggregators.get("composite")
        if not isinstance(composite_name, str):
            return

        from metis.calibrate.optimization.aggregator_tuner import AggregatorTuner

        tuner = AggregatorTuner(logger=logger)
        composite_func = tuner.aggregation_functions.get(composite_name)
        if composite_func is None:
            return

        if not tuner.is_degenerate_composite_aggregator(composite_func):
            return

        upper_metric_data, lower_metric_data = self._metric_details_to_tuner_input(
            self.metric_details
        )
        if not upper_metric_data or not lower_metric_data:
            logger.warning(
                "Cached composite aggregator '%s' is degenerate but metric_details could not be reconstructed",
                composite_name,
            )
            return

        refreshed = tuner.tune_from_metrics(upper_metric_data, lower_metric_data).get("optimal")
        if not isinstance(refreshed, dict):
            return

        refreshed_composite = refreshed.get("composite")
        if refreshed_composite == composite_name:
            return

        logger.info(
            "Refreshing cached composite aggregator from %s to %s using persisted metric details",
            composite_name,
            refreshed_composite,
        )
        self.optimal_aggregators = refreshed
        self.metadata["optimal_aggregators_refreshed"] = True
        self.metadata["optimal_aggregators_refreshed_from"] = composite_name
        self.metadata["optimal_aggregators_refreshed_to"] = refreshed_composite

    @classmethod
    def load(cls, filepath: str) -> "CalibrationBounds":
        """
        Load bounds from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            CalibrationBounds instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {filepath}\n"
                f"You must run calibration first: python -m metis calibrate ..."
            )

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Calibration file {filepath!s} must contain a JSON object at the "
                f"top level, got {type(data).__name__}."
            )
        bounds = data.get("bounds", {})
        if not isinstance(bounds, dict):
            raise ValueError(
                f"Calibration file {filepath!s}: 'bounds' must be a JSON object, "
                f"got {type(bounds).__name__}."
            )
        for family, fam_bounds in bounds.items():
            if not isinstance(fam_bounds, dict):
                raise ValueError(
                    f"Calibration file {filepath!s}: bounds[{family!r}] must be "
                    f"a JSON object of metric→bounds, got {type(fam_bounds).__name__}."
                )

        instance = cls()
        instance.bounds = bounds
        instance.metadata = data.get("metadata", {}) or {}
        instance.optimal_aggregators = data.get("optimal_aggregators")
        instance.metric_details = data.get("metric_details")
        instance._refresh_optimal_aggregators_from_metric_details()

        # Validate that at least one family has bounds
        if not instance.bounds:
            raise ValueError(
                "Calibration file contains no bounds data.\nPlease re-run calibration."
            )

        return instance

    def set_metadata(self, key: str, value: Any) -> None:
        """set metadata field."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata field with optional default.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)

    def validate_against(self, data: pd.DataFrame, config_path: str | None = None) -> bool:
        """
        Validate if these bounds are valid for given data and config.

        Checks:
        1. Data fingerprint match (if data provided)
        2. Config fingerprint match (if config_path provided)
        3. Data shape match (basic sanity check)

        Args:
            data: DataFrame to validate against
            config_path: Path to config YAML to validate against

        Returns:
            True if bounds are valid for this data/config, False otherwise

        Examples:
            >>> bounds = CalibrationBounds.load("bounds.json")
            >>> is_valid = bounds.validate_against(my_data, "config.yaml")
            >>> if not is_valid:
            ...     print("Need to recalibrate!")
        """
        # Check if metadata has fingerprints
        stored_data_fp = self.get_metadata("data_fingerprint")
        stored_config_fp = self.get_metadata("config_fingerprint")

        if not stored_data_fp:
            logger.warning(
                "Calibration bounds missing data_fingerprint. Cannot validate - assume invalid."
            )
            return False

        # Validate data fingerprint
        from metis.calibrate.cache.fingerprint import compute_data_fingerprint

        current_data_fp = compute_data_fingerprint(data)
        if current_data_fp != stored_data_fp:
            logger.warning(
                f"Data fingerprint mismatch:\n"
                f"  Stored:  {stored_data_fp[:16]}...\n"
                f"  Current: {current_data_fp[:16]}..."
            )
            return False

        # Validate config fingerprint if provided
        if config_path and stored_config_fp:
            from metis.calibrate.cache.fingerprint import compute_config_fingerprint

            current_config_fp = compute_config_fingerprint(config_path)
            if current_config_fp != stored_config_fp:
                logger.warning(
                    f"Config fingerprint mismatch:\n"
                    f"  Stored:  {stored_config_fp}\n"
                    f"  Current: {current_config_fp}"
                )
                return False

        # Basic sanity check: shape
        stored_shape = self.get_metadata("dataset_shape")
        if stored_shape and list(data.shape) != stored_shape:
            logger.warning(
                f"Dataset shape mismatch:\n  Stored:  {stored_shape}\n  Current: {list(data.shape)}"
            )
            return False

        logger.info("Calibration bounds validated successfully")
        return True

    def get_summary(self) -> str:
        """
        Get human-readable summary of bounds.

        Returns:
            Formatted string summary
        """
        lines = ["=" * 60, "CALIBRATION BOUNDS SUMMARY", "=" * 60]

        for family in ["fidelity", "privacy", "utility"]:
            if family in self.bounds:
                b = self.bounds[family]
                lines.append(f"\n{family.upper()}:")
                lines.append(f"  Lower bound (real vs noise): {b['lower']:.4f}")
                lines.append(f"  Upper bound (real vs real):  {b['upper']:.4f}")
                lines.append(f"  Range: [{b['lower']:.4f}, {b['upper']:.4f}]")
                if b.get("lower_iterations"):
                    lines.append(f"  Iterations: {len(b['lower_iterations'])}")

        if self.optimal_aggregators:
            lines.append("\nOPTIMAL AGGREGATORS:")
            for key, value in self.optimal_aggregators.items():
                lines.append(f"  {key}: {value}")

        lines.append("\nMetadata:")
        for key, value in self.metadata.items():
            lines.append(f"  {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "bounds": self.bounds,
            "metadata": self.metadata,
        }
        if self.optimal_aggregators:
            d["optimal_aggregators"] = self.optimal_aggregators
        if self.metric_details:
            d["metric_details"] = self.metric_details
        return d

    def __len__(self) -> int:
        """Return number of families with bounds."""
        return len(self.bounds)

    def __contains__(self, family: str) -> bool:
        """Check if family has bounds."""
        return family in self.bounds

    def __repr__(self) -> str:
        return f"CalibrationBounds(families={list(self.bounds.keys())})"
