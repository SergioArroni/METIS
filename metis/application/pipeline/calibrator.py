"""Step 4 — Calibrate evaluation bounds (or load from cache).

Single responsibility: decide whether calibration is needed, run it or
load cached results, and return CalibrationBounds (or None).
"""

from pathlib import Path
from typing import Any

import pandas as pd

from metis.calibrate import CalibrationBounds
from metis.calibrate.cache.cache_manager import CacheManager

from ...domain.errors import ConfigError
from ...infrastructure.runtime.logging import get_logger

# Defaults shared by the pipeline step and ``metis calibrate``.
CALIBRATION_DEFAULTS: dict[str, Any] = {
    "n_iterations": 10,
    "sample_percentage": 100.0,
    "n_jobs": 1,
    # Ignored since calibration is per metric (kept so old configs validate).
    "tune_aggregators": False,
    "force": False,
}


def calibration_params(config: dict[str, Any], n_iterations: int | None = None) -> dict[str, Any]:
    """Resolve calibration parameters from *config* with the shared defaults.

    Seed priority: ``calibration.base_seed`` → ``reproducibility.seed`` → 42.
    ``n_iterations`` (e.g. ``metis calibrate -n``) overrides the config.
    """
    cal_cfg = config.get("calibration") or {}
    repro_seed = (config.get("reproducibility") or {}).get("seed", 42)
    return {
        "n_iterations": n_iterations
        or cal_cfg.get("n_iterations", CALIBRATION_DEFAULTS["n_iterations"]),
        "sample_percentage": cal_cfg.get(
            "sample_percentage", CALIBRATION_DEFAULTS["sample_percentage"]
        ),
        "base_seed": cal_cfg.get("base_seed", repro_seed),
        "n_jobs": cal_cfg.get("n_jobs", CALIBRATION_DEFAULTS["n_jobs"]),
        "force_recalibrate": cal_cfg.get("force", CALIBRATION_DEFAULTS["force"]),
        "tune_aggregators": cal_cfg.get(
            "tune_aggregators", CALIBRATION_DEFAULTS["tune_aggregators"]
        ),
    }


class CalibrationStep:
    """Produces or loads ``CalibrationBounds`` for the current dataset."""

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    def calibrate_if_needed(
        self,
        real_data: pd.DataFrame | None,
        config: dict[str, Any],
        config_path: str | None = None,
    ) -> CalibrationBounds | None:
        """Return calibration bounds, running calibration if necessary.

        Decision logic:
          1. If ``calibration.mode == "default"`` → skip (use median aggregation).
          2. If ``calibration.enabled == False`` → skip, return None.
          3. If ``calibration.bounds_file`` specified → load from that path.
          4. Otherwise → auto-discover from cache or run new calibration.
          5. If ``calibration.force == True`` → always recalibrate.

        Returns:
            ``CalibrationBounds`` instance, or ``None`` when calibration is
            disabled / not configured.
        """
        cal_cfg = config.get("calibration") or {}

        # 1. Check if default mode (no calibration, use median)
        if cal_cfg.get("mode") == "default":
            self._logger.info("Calibration mode set to 'default' — using median aggregation")
            return None

        # 2. Calibration explicitly disabled
        if not cal_cfg or cal_cfg.get("enabled") is False:
            self._logger.info("Calibration disabled or not configured — skipping")
            return None

        # 3. Check for explicit bounds_file path
        bounds_path = cal_cfg.get("bounds_file")
        if bounds_path:
            bounds_file = Path(bounds_path)
            if not bounds_file.exists():
                self._logger.error(
                    "Specified bounds_file not found: %s\n"
                    "Remove 'bounds_file' from config to auto-discover or create new bounds.",
                    bounds_path,
                )
                raise FileNotFoundError(f"Calibration file not found: {bounds_path}")

            self._logger.info("Loading calibration bounds from %s", bounds_path)
            return CalibrationBounds.load(bounds_path)

        # 4. Auto-discover from cache or run new calibration
        if config_path is None:
            self._logger.warning(
                "Config path not available — cannot run automatic calibration. "
                "Specify 'calibration.bounds_file' or provide config_path."
            )
            return None

        return self.run_calibration(real_data, config, config_path)

    def run_calibration(
        self,
        real_data: pd.DataFrame | None,
        config: dict[str, Any],
        config_path: str,
        n_iterations: int | None = None,
    ) -> CalibrationBounds:
        """Calibrate (or load from cache) unconditionally.

        Used by the pipeline and by ``metis calibrate`` so both share the
        defaults (:data:`CALIBRATION_DEFAULTS`), the seed priority, the
        cache name and the data: *real_data* must be the real table cleaned
        but **not cast** (``ReferenceContext.real_clean``): every calibration
        iteration runs it through the full pipeline, casting included.

        ``calibration.cache_dir`` sets where the bounds are cached (default
        ``metis/calibrate/cache``, relative to the working directory).
        """
        if real_data is None:
            raise ConfigError(
                "Calibration needs the real table before casting "
                "(ReferenceContext.real_clean); build the reference with prepare_reference()"
            )
        params = calibration_params(config, n_iterations)
        if params["force_recalibrate"]:
            self._logger.info("Force recalibration enabled — ignoring cache")

        # Extract dataset name from real data path for cache filename
        data_cfg = config.get("data", {})
        dataset_name = self._extract_dataset_name(data_cfg.get("real", ""))

        cache_kwargs: dict[str, Any] = {"dataset_name": dataset_name}
        cache_dir = (config.get("calibration") or {}).get("cache_dir")
        if cache_dir:
            cache_kwargs["cache_dir"] = str(cache_dir)
        cache_mgr = CacheManager(**cache_kwargs)
        bounds = cache_mgr.get_or_calibrate(real_data=real_data, config_path=config_path, **params)
        self._logger.info("Calibration complete")
        return bounds

    @staticmethod
    def _extract_dataset_name(real_path: str) -> str | None:
        """Extract a short dataset name from the real data CSV path.

        Examples:
            'data/real/telco.csv' → 'telco'
            'data/real/hiperam_clean.csv' → 'hiperam_clean'
        """
        if not real_path:
            return None
        return Path(real_path).stem
