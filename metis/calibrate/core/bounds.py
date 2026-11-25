"""
Per-metric calibration bounds: storage, normalisation and loading.

Every calibrated metric gets its own pair of empirical anchors, measured over
``n`` iterations (study ``calib_v2``, 2026-10-09, approved by the author):

* **fidelity** — upper: real rows the generator never saw (a third of the
  table) against its training third; lower: uniform noise.
* **privacy** — upper: the same unseen real third; lower: an exact copy of
  the training third.  Both measured *with* the holdout third, as the
  evaluation is (without it the copy and the noise both score ≈1 and the
  bounds come out inverted).
* **utility** — not calibrated: its absolute scale ``(TSTR − T)/(TTR − T)``
  already anchors 0 (no signal) and 1 (as useful as the real table).  The
  calibration only decides, once per real table, whether the target has
  signal measurable at its size (``utility_signal``,
  ``metis.calibrate.core.utility_signal``); every evaluation with these
  bounds uses that decision and, on tables of the size of a third, its
  anchors, instead of its own five half-samples.

A metric score ``s`` is calibrated as ``clip((s − l) / (u − l), 0, 1)`` and
then aggregated through the normal hierarchy.  ``l``/``u`` are the one-sided
95 % quantiles of the iterations towards each other (``mean_l + z·sd_l``,
``mean_u − z·sd_u``; ``metis.shared.anchors``): an unseen real sample scores
1 and noise 0, instead of scattering around them and averaging below 1 and
above 0 once clipped (benchmark of 2026-10-10).  A metric whose range
(between the means) is below ``NOISE_MULTIPLE`` times its noise (the
iterations' standard deviations combined, ``sqrt(sd_l² + sd_u²)``) cannot
tell its anchors apart: it is excluded from the aggregation instead of
amplifying noise.  The factor 5 is
the criterion of the study (``calib_v2``): with a range of five noise units
the two anchors' iteration clouds do not overlap, so the rescaling reflects
the metric and not the sampling noise of its anchors.

Calibration is per metric only, never per family or category: one method for
the whole of METIS.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metis.shared.anchors import ANCHOR_Z, NOISE_MULTIPLE, WARNING_SD_FLOOR

logger = logging.getLogger(__name__)

# Version of the file layout.  Format 2 (anchors at the means) is rebuilt on
# load from its stored iterations; older family-level files are rejected.
BOUNDS_FORMAT = 3
_MIGRATABLE_FORMATS = (2,)

# Reference descriptions stored with the bounds.
REFERENCES: dict[str, dict[str, str]] = {
    "fidelity": {
        "upper": "unseen real third vs the generator's training third (holdout given)",
        "lower": "uniform noise vs the training third (holdout given)",
    },
    "privacy": {
        "upper": "unseen real third vs the generator's training third (holdout given)",
        "lower": "exact copy of the training third (holdout given)",
    },
}
# Families that are calibrated.  Utility is excluded on purpose (module docstring).
CALIBRATED_FAMILIES: tuple[str, ...] = tuple(REFERENCES)


def _summary(values: list[float]) -> tuple[float, float]:
    """``(mean, sd)`` of finite *values*; sd is 0 with fewer than two."""
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return float(np.mean(finite)), sd


class CalibrationBounds:
    """Per-metric calibration bounds (see the module docstring)."""

    def __init__(self):
        """Initialize empty bounds storage."""
        self.bounds: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}
        # Kept for callers that read it: per-metric calibration fixes the
        # anchors itself, so no aggregation function is tuned any more.
        self.optimal_aggregators: dict[str, Any] | None = None
        # Whether the real table has signal for the utility metrics (None:
        # not measured, each evaluation decides on its own).
        self.utility_signal: dict[str, Any] | None = None

    def set_bounds(
        self,
        metric_id: str,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        lower_iterations: list[float] | None = None,
        upper_iterations: list[float] | None = None,
        *,
        family: str | None = None,
        reference_lower: str | None = None,
        reference_upper: str | None = None,
    ) -> None:
        """Store the anchors of *metric_id*.

        With iterations the anchors are their one-sided quantiles towards
        each other (``lower_bound`` and ``upper_bound`` are then ignored) and
        the noise is ``sqrt(sd_lower² + sd_upper²)``.  Without iterations the
        given bounds are used with zero noise.
        """
        lower_iterations = [float(x) for x in lower_iterations or []]
        upper_iterations = [float(x) for x in upper_iterations or []]
        if lower_iterations:
            lower, lower_sd = _summary(lower_iterations)
        else:
            lower, lower_sd = float(lower_bound), 0.0
        if upper_iterations:
            upper, upper_sd = _summary(upper_iterations)
        else:
            upper, upper_sd = float(upper_bound), 0.0
        if family is None:
            from metis.domain.taxonomy import get_metric_hierarchy

            family = get_metric_hierarchy(metric_id)["family"]

        noise = float(np.hypot(lower_sd, upper_sd))
        span = upper - lower
        lower_mean, upper_mean = lower, upper
        if not (np.isfinite(lower) and np.isfinite(upper)):
            calibratable, reason = False, "excluded: a reference has no finite value"
        elif span <= 0:
            calibratable, reason = (
                False,
                f"excluded: upper reference ({upper:.4f}) not above the lower one ({lower:.4f})",
            )
        elif span < NOISE_MULTIPLE * noise:
            calibratable, reason = (
                False,
                f"excluded: range < {NOISE_MULTIPLE:g}×noise (range {span:.4f}, noise {noise:.4f})",
            )
        else:
            calibratable, reason = True, None
            lower = lower_mean + ANCHOR_Z * lower_sd
            upper = upper_mean - ANCHOR_Z * upper_sd

        self.bounds[metric_id] = {
            "family": family,
            "lower": lower,
            "upper": upper,
            "lower_mean": lower_mean,
            "upper_mean": upper_mean,
            "lower_sd": lower_sd,
            "upper_sd": upper_sd,
            "noise": noise,
            "n_lower": len(lower_iterations),
            "n_upper": len(upper_iterations),
            "lower_iterations": lower_iterations,
            "upper_iterations": upper_iterations,
            "reference_lower": reference_lower,
            "reference_upper": reference_upper,
            "calibratable": calibratable,
            "reason": reason,
        }

    def get_bounds(self, metric_id: str) -> tuple[float, float]:
        """``(lower, upper)`` anchors of *metric_id*."""
        if metric_id not in self.bounds:
            raise KeyError(f"Metric '{metric_id}' not found in calibration bounds")
        data = self.bounds[metric_id]
        return (data["lower"], data["upper"])

    def is_calibratable(self, metric_id: str) -> bool:
        return bool(self.bounds.get(metric_id, {}).get("calibratable", False))

    def exclusion_reason(self, metric_id: str) -> str | None:
        return self.bounds.get(metric_id, {}).get("reason")

    def get_all_metrics(self) -> list[str]:
        return sorted(self.bounds)

    def get_all_families(self) -> list[str]:
        """Families with at least one calibrated metric."""
        return sorted({b["family"] for b in self.bounds.values()})

    def normalize_with_bounds(self, metric_id: str, raw_value: float) -> float:
        """``clip((raw − lower) / (upper − lower), 0, 1)``; NaN stays NaN.

        Raises ``KeyError`` for an unknown metric and ``ValueError`` for one
        excluded from calibration (see ``is_calibratable``).
        """
        if metric_id not in self.bounds:
            raise KeyError(f"No calibration bounds for metric '{metric_id}'")
        if not np.isfinite(raw_value):
            return float("nan")
        entry = self.bounds[metric_id]
        if not entry["calibratable"]:
            raise ValueError(f"{metric_id}: {entry['reason']}")
        lower, upper = entry["lower"], entry["upper"]
        return float(np.clip((raw_value - lower) / (upper - lower), 0.0, 1.0))

    def deviation_from_real(self, metric_id: str, raw_value: float) -> float | None:
        """How many standard deviations of another real sample *raw_value* falls below it.

        ``(upper_mean − raw) / max(upper_sd, WARNING_SD_FLOOR)``: the upper
        reference is the unseen real sample, so a value above ``WARNING_Z``
        is a metric worse than real sampling noise explains.  ``None`` for an
        unknown or excluded metric, a non-finite value or an entry without
        iterations of its upper reference.
        """
        entry = self.bounds.get(metric_id)
        if not entry or not entry.get("calibratable") or not np.isfinite(raw_value):
            return None
        if not entry.get("n_upper") or entry.get("upper_mean") is None:
            return None
        sd = max(float(entry.get("upper_sd") or 0.0), WARNING_SD_FLOOR)
        return float((entry["upper_mean"] - raw_value) / sd)

    def save(self, filepath: str) -> None:
        """Save bounds to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata["saved_at"] = datetime.now().isoformat()
        if "data_fingerprint" not in self.metadata:
            logger.warning(
                "Saving calibration bounds without data_fingerprint. "
                "Cache validation will not work properly."
            )
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> CalibrationBounds:
        """Load bounds from a JSON file written by :meth:`save`.

        Format-2 files (anchors at the iteration means) are rebuilt from their
        stored iterations.  Raises ``ValueError`` for files of the old
        family-level layout: their anchors were measured differently (no
        holdout, noise as the worst privacy case) and cannot be reused.
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
        if data.get("format") in _MIGRATABLE_FORMATS:
            return cls._rebuilt(data)
        if data.get("format") != BOUNDS_FORMAT:
            raise ValueError(
                f"Calibration file {filepath!s} has the old family-level layout "
                f"(format {data.get('format')!r}, expected {BOUNDS_FORMAT}); "
                "re-run the calibration."
            )
        bounds = data.get("bounds", {})
        if not isinstance(bounds, dict) or not bounds:
            raise ValueError(f"Calibration file {filepath!s} contains no bounds; re-run it.")
        required = {"family", "lower", "upper", "calibratable"}
        for metric_id, entry in bounds.items():
            if not isinstance(entry, dict) or not required <= set(entry):
                raise ValueError(
                    f"Calibration file {filepath!s}: bounds[{metric_id!r}] must hold "
                    f"{sorted(required)}"
                )
        instance = cls()
        instance.bounds = bounds
        instance.metadata = data.get("metadata", {}) or {}
        instance.utility_signal = data.get("utility_signal")
        return instance

    @classmethod
    def _rebuilt(cls, data: dict[str, Any]) -> CalibrationBounds:
        """Bounds of a format-2 file, anchors recomputed from its iterations."""
        instance = cls()
        for metric_id, entry in (data.get("bounds") or {}).items():
            instance.set_bounds(
                metric_id,
                lower_bound=entry.get("lower"),
                upper_bound=entry.get("upper"),
                lower_iterations=entry.get("lower_iterations"),
                upper_iterations=entry.get("upper_iterations"),
                family=entry.get("family"),
                reference_lower=entry.get("reference_lower"),
                reference_upper=entry.get("reference_upper"),
            )
        if not instance.bounds:
            raise ValueError("Calibration file contains no bounds; re-run it.")
        instance.metadata = dict(data.get("metadata") or {})
        instance.metadata["rebuilt_from_format"] = data.get("format")
        instance.utility_signal = data.get("utility_signal")
        return instance

    def set_metadata(self, key: str, value: Any) -> None:
        """set metadata field."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Metadata field, or *default*."""
        return self.metadata.get(key, default)

    def validate_against(self, data: pd.DataFrame, config_path: str | None = None) -> bool:
        """True if these bounds were calibrated on *data* (and *config_path*)."""
        stored_data_fp = self.get_metadata("data_fingerprint")
        stored_config_fp = self.get_metadata("config_fingerprint")
        if not stored_data_fp:
            logger.warning(
                "Calibration bounds missing data_fingerprint. Cannot validate - assume invalid."
            )
            return False

        from metis.calibrate.cache.fingerprint import compute_data_fingerprint

        current_data_fp = compute_data_fingerprint(data)
        if current_data_fp != stored_data_fp:
            logger.warning(
                "Data fingerprint mismatch:\n  Stored:  %s...\n  Current: %s...",
                stored_data_fp[:16],
                current_data_fp[:16],
            )
            return False
        if config_path and stored_config_fp:
            from metis.calibrate.cache.fingerprint import compute_config_fingerprint

            current_config_fp = compute_config_fingerprint(config_path)
            if current_config_fp != stored_config_fp:
                logger.warning(
                    "Config fingerprint mismatch:\n  Stored:  %s\n  Current: %s",
                    stored_config_fp,
                    current_config_fp,
                )
                return False
        stored_shape = self.get_metadata("dataset_shape")
        if stored_shape and list(data.shape) != stored_shape:
            logger.warning(
                "Dataset shape mismatch:\n  Stored:  %s\n  Current: %s",
                stored_shape,
                list(data.shape),
            )
            return False
        logger.info("Calibration bounds validated successfully")
        return True

    def get_summary(self) -> str:
        """Human-readable summary, one line per metric."""
        lines = ["=" * 60, "CALIBRATION BOUNDS (per metric)", "=" * 60]
        for family in self.get_all_families():
            lines.append(f"\n{family.upper()}:")
            for metric_id in self.get_all_metrics():
                b = self.bounds[metric_id]
                if b["family"] != family:
                    continue
                status = "ok" if b["calibratable"] else b["reason"]
                lines.append(
                    f"  {metric_id}: [{b['lower']:.4f} ± {b['lower_sd']:.4f}, "
                    f"{b['upper']:.4f} ± {b['upper_sd']:.4f}] {status}"
                )
        if self.utility_signal is not None:
            u = self.utility_signal
            lines.append(
                f"\nUTILITY SIGNAL: {'applicable' if u['applicable'] else 'not applicable'} "
                f"(TTR gain {u['ttr_gain']:.4f}, threshold {u['threshold']:.4f}, "
                f"{u['n_half_samples']} half-samples)"
            )
        lines.append("\nMetadata:")
        for key, value in self.metadata.items():
            lines.append(f"  {key}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {"format": BOUNDS_FORMAT, "bounds": self.bounds, "metadata": self.metadata}
        if self.utility_signal is not None:
            data["utility_signal"] = self.utility_signal
        return data

    def __len__(self) -> int:
        """Number of metrics with bounds."""
        return len(self.bounds)

    def __contains__(self, metric_id: str) -> bool:
        return metric_id in self.bounds

    def __repr__(self) -> str:
        return f"CalibrationBounds(metrics={len(self.bounds)}, families={self.get_all_families()})"
