"""Smart cache manager for calibration results with automatic validation."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from metis.calibrate.cache.fingerprint import (
    compute_config_fingerprint,
    compute_data_fingerprint,
    generate_cache_key,
)
from metis.calibrate.core.bounds import CalibrationBounds
from metis.calibrate.core.calibrator import MetricCalibrator

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages calibration result caching with fingerprint-based validation.

    Automatically detects when cached calibration results can be reused based on:
    - Data content fingerprint (1000-row sample)
    - Configuration fingerprint (relevant metrics/schema)
    - Calibration parameters (iterations, sample_size, seed)

    Usage:
        >>> manager = CacheManager(cache_dir="metis/calibrate/cache")
        >>> bounds = manager.get_or_calibrate(
        ...     real_data=df, config_path="config.yaml", n_iterations=12, sample_size=750
        ... )
        # Returns cached if valid, else runs calibration
    """

    def __init__(self, cache_dir: str = "metis/calibrate/cache", dataset_name: str = None):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cached calibration results
            dataset_name: Name of the dataset (e.g., 'cardio', 'airbnb', 'telco')
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name

    def _get_cache_path(self, cache_key: str, date_tag: str | None = None) -> Path:
        """Get full path for cache file.

        Args:
            cache_key: Unique cache key from fingerprints + params.
            date_tag: Optional date string (YYYYMMDD) appended to filename.
                      When ``None``, today's date is used for *new* files;
                      for lookups, :meth:`_find_cache_path` is preferred.
        """
        tag = date_tag or datetime.now().strftime("%Y%m%d")
        if self.dataset_name:
            filename = f"bounds_{self.dataset_name}_{cache_key}_{tag}.json"
        else:
            filename = f"bounds_{cache_key}_{tag}.json"
        # Safety: cap filename at 150 chars to stay within Windows MAX_PATH
        if len(filename) > 150:
            import hashlib

            suffix = hashlib.sha256(filename.encode()).hexdigest()[:16]
            filename = f"bounds_{self.dataset_name or 'cal'}_{suffix}_{tag}.json"
        return self.cache_dir / filename

    def _find_cache_path(self, cache_key: str) -> Path | None:
        """Find an existing cache file for *cache_key* (any date tag).

        Returns the most recent match, or ``None`` if nothing found.
        """
        if self.dataset_name:
            pattern = f"bounds_{self.dataset_name}_{cache_key}_*.json"
        else:
            pattern = f"bounds_{cache_key}_*.json"

        matches = sorted(self.cache_dir.glob(pattern))
        return matches[-1] if matches else None

    def find_cache_path(self, cache_key: str) -> Path | None:
        """Public accessor for the latest cache file matching *cache_key*.

        Mirrors :meth:`_find_cache_path` and exists so external callers
        (e.g. the SOTA benchmark orchestrator) do not have to reach into
        protected members of this class.
        """
        return self._find_cache_path(cache_key)

    def _get_aggregators_cache_path(self, cache_key: str) -> Path:
        """Get full path for optimal aggregators cache file."""
        return self.cache_dir / f"aggregators_{cache_key}.json"

    def get_or_calibrate(
        self,
        real_data: pd.DataFrame,
        config_path: str,
        n_iterations: int = 12,
        sample_percentage: float = 100.0,  # Percentage of dataset to use
        base_seed: int = 42,
        n_jobs: int = -1,
        force_recalibrate: bool = False,
        tune_aggregators: bool = True,
    ) -> CalibrationBounds:
        """
        Get cached calibration or run new calibration if cache invalid.

        Automatically validates cache based on:
        1. Data fingerprint match
        2. Config fingerprint match
        3. Parameter match (iterations, sample_percentage, seed)

        Args:
            real_data: Real dataset to calibrate against
            config_path: Path to YAML configuration
            n_iterations: Number of calibration iterations
            sample_percentage: Percentage of dataset to use (100.0 = all data)
            base_seed: Base random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            force_recalibrate: Force recalibration even if cache valid

        Returns:
            CalibrationBounds object (cached or newly computed)

        Logs:
            - Info on cache hit (with date and fingerprint)
            - Warning on cache miss reason
            - Info on new calibration start
        """
        # Convert percentage to actual sample size
        sample_size = int(len(real_data) * sample_percentage / 100.0)

        # Compute fingerprints
        logger.info("Computing data and config fingerprints...")
        data_fp = compute_data_fingerprint(real_data)
        config_fp = compute_config_fingerprint(config_path)

        # Generate cache key
        cache_key = generate_cache_key(
            data_fingerprint=data_fp,
            config_fingerprint=config_fp,
            n_iterations=n_iterations,
            sample_size=sample_size,
            base_seed=base_seed,
        )

        # Look for any existing cache file for this key (regardless of date)
        cache_path = self._find_cache_path(cache_key)

        # Check if cache exists and is valid
        if not force_recalibrate and cache_path is not None and cache_path.exists():
            logger.info("Found cached calibration: %s", cache_path.name)

            try:
                bounds = CalibrationBounds.load(str(cache_path))

                # Validate fingerprints
                cached_data_fp = bounds.get_metadata("data_fingerprint")
                cached_config_fp = bounds.get_metadata("config_fingerprint")

                if cached_data_fp == data_fp and cached_config_fp == config_fp:
                    calibration_date = bounds.get_metadata("calibration_date", "unknown")
                    logger.info(
                        "Cache hit! Using calibration from %s\n"
                        "  Data fingerprint: %s...\n"
                        "  Config fingerprint: %s",
                        calibration_date,
                        data_fp[:16],
                        config_fp,
                    )
                    return bounds
                logger.warning(
                    "Cache invalid - fingerprint mismatch:\n"
                    "  Data: %s... vs %s...\n"
                    "  Config: %s vs %s",
                    cached_data_fp[:16],
                    data_fp[:16],
                    cached_config_fp,
                    config_fp,
                )
            except Exception as e:
                logger.warning("Failed to load cache: %s", e)

        # Cache miss or invalid - run calibration
        logger.info(
            "Running new calibration:\n"
            "  Iterations: %d\n"
            "  Sample percentage: %.1f%% (%d rows)\n"
            "  Base seed: %d\n"
            "  Jobs: %d",
            n_iterations,
            sample_percentage,
            sample_size,
            base_seed,
            n_jobs,
        )

        calibrator = MetricCalibrator(logger=logger)
        bounds = calibrator.calibrate(
            real_data=real_data,
            config_template_path=config_path,
            n_iterations=n_iterations,
            sample_size=sample_size,
            base_seed=base_seed,
            n_jobs=n_jobs,
            tune_aggregators=tune_aggregators,
        )

        # Add fingerprints and full parameters to metadata before saving
        bounds.set_metadata("data_fingerprint", data_fp)
        bounds.set_metadata("config_fingerprint", config_fp)
        bounds.set_metadata("cache_key", cache_key)
        bounds.set_metadata("calibration_date", datetime.now().strftime("%Y-%m-%d"))
        bounds.set_metadata("config_path", config_path)
        if self.dataset_name:
            bounds.set_metadata("dataset_name", self.dataset_name)

        # Store the metric list used for calibration
        import yaml

        with Path(config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        metric_ids = cfg.get("metrics") or cfg.get("evaluation", {}).get("metric_ids", [])
        bounds.set_metadata("metric_ids", metric_ids)
        bounds.set_metadata("task_type", cfg.get("data", {}).get("task_type"))

        # Save to cache (filename includes today's date)
        cache_path = self._get_cache_path(cache_key)
        bounds.save(str(cache_path))
        logger.info("Calibration complete and cached: %s", cache_path.name)

        return bounds

    def invalidate_cache(self, cache_key: str | None = None) -> None:
        """
        Invalidate cached calibration results.

        Args:
            cache_key: Specific cache key to invalidate. If None, clears all cache.
        """
        if cache_key:
            cache_path = self._get_cache_path(cache_key)
            agg_path = self._get_aggregators_cache_path(cache_key)

            if cache_path.exists():
                cache_path.unlink()
                logger.info("Invalidated cache: %s", cache_key)

            if agg_path.exists():
                agg_path.unlink()
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared all calibration cache")

    def list_cache(self) -> list:
        """
        list all cached calibration results with metadata.

        Returns:
            list of dicts with cache information
        """
        cache_list = []

        for cache_file in sorted(self.cache_dir.glob("bounds_*.json")):
            try:
                bounds = CalibrationBounds.load(str(cache_file))
                cache_list.append(
                    {
                        "file": cache_file.name,
                        "dataset_name": bounds.get_metadata("dataset_name", "unknown"),
                        "cache_key": bounds.get_metadata("cache_key", "unknown"),
                        "date": bounds.get_metadata("calibration_date", "unknown"),
                        "data_fingerprint": bounds.get_metadata("data_fingerprint", "unknown")[:16]
                        + "...",
                        "n_iterations": bounds.get_metadata("n_iterations", "unknown"),
                        "sample_size": bounds.get_metadata("sample_size", "unknown"),
                    }
                )
            except Exception as e:
                logger.warning("Failed to read cache %s: %s", cache_file.name, e)

        return cache_list
