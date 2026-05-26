"""Step 4 — Calibrate evaluation bounds (or load from cache).

Single responsibility: decide whether calibration is needed, run it or
load cached results, and return CalibrationBounds (or None).
"""

from pathlib import Path
from typing import Any

import pandas as pd

from metis.calibrate import CalibrationBounds
from metis.calibrate.cache.cache_manager import CacheManager

from ...infrastructure.runtime.logging import get_logger


class CalibrationStep:
    """Produces or loads ``CalibrationBounds`` for the current dataset."""

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    def calibrate_if_needed(
        self,
        real_data: pd.DataFrame,
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
        cal_cfg = config.get("calibration", {})

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

        force_recalibrate = cal_cfg.get("force", False)
        if force_recalibrate:
            self._logger.info("Force recalibration enabled — ignoring cache")

        # Seed priority: calibration.base_seed → reproducibility.seed → 42
        repro_seed = config.get("reproducibility", {}).get("seed", 42)
        base_seed = cal_cfg.get("base_seed", repro_seed)

        # Extract dataset name from real data path for cache filename
        data_cfg = config.get("data", {})
        dataset_name = self._extract_dataset_name(data_cfg.get("real", ""))

        cache_mgr = CacheManager(dataset_name=dataset_name)
        bounds = cache_mgr.get_or_calibrate(
            real_data=real_data,
            config_path=config_path,
            n_iterations=cal_cfg.get("n_iterations", 10),
            sample_percentage=cal_cfg.get("sample_percentage", 100.0),
            base_seed=base_seed,
            n_jobs=cal_cfg.get("n_jobs", 1),
            force_recalibrate=force_recalibrate,
            tune_aggregators=cal_cfg.get("tune_aggregators", True),
        )
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
