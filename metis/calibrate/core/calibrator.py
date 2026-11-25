"""Main calibrator: per-metric bounds from an upper and a lower reference.

Flow (see ``metis.calibrate.core.bounds`` for the method and its rationale):

1. Upper strategy: ``n`` iterations of an unseen real third against the
   training third, holdout third given (fidelity and privacy metrics).
2. Lower strategy: the same thirds with uniform noise (fidelity metrics) and
   an exact copy of the training third (privacy metrics).
3. Per metric: anchors = one-sided 95 % quantiles of the iterations
   towards each other (``metis.shared.anchors``), noise = their standard
   deviations combined; metrics whose range is below ``NOISE_MULTIPLE``
   times their noise are stored as excluded.

Utility is never calibrated; the calibration only decides once whether the
real table has signal for it (``metis.calibrate.core.utility_signal``).
Nothing is calibrated per family or category.  The aggregation functions are not tuned any more: the per-metric
anchors already put every metric on the same 0-1 scale, and the evaluation's
hierarchy (median, then SSD) is the one method of METIS.

Uses Strategy Pattern to delegate to specialized strategies
and Dependency Injection to decouple from concrete implementations.
"""

import logging
import time
import warnings
from typing import Any

import pandas as pd

from metis.calibrate.core.bounds import (
    CALIBRATED_FAMILIES,
    NOISE_MULTIPLE,
    REFERENCES,
    CalibrationBounds,
)


class MetricCalibrator:
    """Per-metric calibration orchestrator (upper and lower reference strategies)."""

    def __init__(
        self,
        upper_strategy=None,
        lower_strategy=None,
        bounds_storage=None,
        logger: logging.Logger | None = None,
        utility_probe=None,
    ):
        """``utility_probe(real_data, config_template_path, sample_size,
        base_seed)`` decides whether the real table has signal for the utility
        metrics (``metis.calibrate.core.utility_signal``); it defaults to the
        real probe only when the strategies are the default ones too."""
        self.logger = logger or logging.getLogger(__name__)
        if utility_probe is None and upper_strategy is None and lower_strategy is None:
            utility_probe = self._probe_utility_signal
        self.utility_probe = utility_probe

        # Lazy defaults: only created if not injected
        if upper_strategy is None or lower_strategy is None:
            from metis.calibrate.strategies import LowerBoundStrategy, UpperBoundStrategy
            from metis.calibrate.utils import InMemoryEvaluator, UniformNoiseGenerator

            evaluator = InMemoryEvaluator
            if upper_strategy is None:
                upper_strategy = UpperBoundStrategy(evaluator=evaluator, logger=self.logger)
            if lower_strategy is None:
                lower_strategy = LowerBoundStrategy(
                    evaluator=evaluator,
                    noise_generator=UniformNoiseGenerator(),
                    logger=self.logger,
                )

        self.upper_strategy = upper_strategy
        self.lower_strategy = lower_strategy
        self.bounds = bounds_storage or CalibrationBounds()

    def calibrate(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        n_iterations: int = 10,
        sample_size: int = 1000,
        base_seed: int = 42,
        n_jobs: int = -1,
        tune_aggregators: bool = False,
    ) -> CalibrationBounds:
        """Run both reference strategies and store per-metric bounds.

        ``tune_aggregators`` is accepted for old configs and ignored (warning):
        the per-metric anchors replace the aggregator tuning.
        """
        if tune_aggregators:
            warnings.warn(
                "calibration.tune_aggregators is ignored: calibration is per metric and "
                "the evaluation's aggregation (median, then SSD) is used as is",
                DeprecationWarning,
                stacklevel=2,
            )
        self.logger.info("=" * 70)
        self.logger.info("METIS CALIBRATION (per metric)")
        self.logger.info("=" * 70)
        self.logger.info("Upper strategy: %s", self.upper_strategy.get_strategy_name())
        self.logger.info("Lower strategy: %s", self.lower_strategy.get_strategy_name())
        self.logger.info(
            "Dataset: %d rows, %d columns; %d iterations, thirds of up to %d rows",
            len(real_data),
            len(real_data.columns),
            n_iterations,
            sample_size,
        )

        start = time.time()
        timings: dict[str, float] = {}
        for label, strategy in (("upper", self.upper_strategy), ("lower", self.lower_strategy)):
            step_start = time.time()
            try:
                strategy.calibrate(
                    real_data=real_data,
                    config_template_path=config_template_path,
                    n_iterations=n_iterations,
                    sample_size=sample_size,
                    base_seed=base_seed,
                    n_jobs=n_jobs,
                )
            except Exception as e:
                # Bounds need both references: with one missing nothing can be
                # calibrated, and empty bounds must not be cached as a success.
                raise RuntimeError(f"{label.capitalize()}-bound calibration failed: {e}") from e
            timings[label] = time.time() - step_start

        self._compute_bounds(
            self.upper_strategy.metric_values_per_iteration,
            self.lower_strategy.metric_values_per_iteration,
        )
        if not self.bounds.get_all_metrics():
            raise RuntimeError(
                "Calibration produced no metric bounds: no fidelity or privacy metric has "
                "values in both references"
            )

        if self.utility_probe is not None:
            step_start = time.time()
            try:
                self.bounds.utility_signal = self.utility_probe(
                    real_data, config_template_path, sample_size, base_seed
                )
            except Exception as e:  # noqa: BLE001 - the evaluations then decide on their own
                self.logger.warning("Utility signal probe failed: %s", e)
            timings["utility_signal"] = time.time() - step_start

        total = time.time() - start
        excluded = {
            m: self.bounds.exclusion_reason(m)
            for m in self.bounds.get_all_metrics()
            if not self.bounds.is_calibratable(m)
        }
        for key, value in {
            "method": "per_metric",
            "references": REFERENCES,
            "noise_multiple": NOISE_MULTIPLE,
            "n_iterations": n_iterations,
            "sample_size": sample_size,
            "base_seed": base_seed,
            "upper_strategy": self.upper_strategy.get_strategy_name(),
            "lower_strategy": self.lower_strategy.get_strategy_name(),
            "calibration_time_seconds": total,
            "upper_time_seconds": timings["upper"],
            "lower_time_seconds": timings["lower"],
            "utility_signal_time_seconds": timings.get("utility_signal"),
            "tune_aggregators": False,
            "excluded_metrics": excluded,
            "dataset_rows": len(real_data),
            "dataset_columns": len(real_data.columns),
        }.items():
            self.bounds.set_metadata(key, value)

        self.logger.info("CALIBRATION COMPLETE in %.1fs", total)
        self.logger.info(self.bounds.get_summary())
        return self.bounds

    def _probe_utility_signal(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        sample_size: int,
        base_seed: int,
    ) -> dict[str, Any] | None:
        from metis.calibrate.core.utility_signal import measure_utility_signal
        from metis.calibrate.utils import InMemoryEvaluator

        evaluator = InMemoryEvaluator(config_template_path)
        from metis.calibrate.strategies.base import config_split_spec

        stratify_by, fractions = config_split_spec(config_template_path)
        return measure_utility_signal(
            evaluator,
            real_data,
            sample_size,
            base_seed,
            self.logger,
            stratify_by=stratify_by,
            fractions=fractions,
        )

    def _compute_bounds(
        self,
        upper_data: dict[str, list[dict[str, float]]],
        lower_data: dict[str, list[dict[str, float]]],
    ) -> None:
        """Per-metric anchors from ``{family: [{metric_id: value}, ...]}`` iterations.

        Only calibrated families (fidelity, privacy) are kept; a metric needs
        values in both references.
        """
        for family in CALIBRATED_FAMILIES:
            upper_iters = upper_data.get(family, [])
            lower_iters = lower_data.get(family, [])
            metric_ids = {m for it in upper_iters for m in it} | {
                m for it in lower_iters for m in it
            }
            refs: dict[str, Any] = REFERENCES[family]
            for metric_id in sorted(metric_ids):
                upper = [it[metric_id] for it in upper_iters if metric_id in it]
                lower = [it[metric_id] for it in lower_iters if metric_id in it]
                if not upper or not lower:
                    self.logger.warning(
                        "%s: no values in the %s reference, not calibrated",
                        metric_id,
                        "upper" if not upper else "lower",
                    )
                    continue
                self.bounds.set_bounds(
                    metric_id,
                    lower_iterations=lower,
                    upper_iterations=upper,
                    family=family,
                    reference_lower=refs["lower"],
                    reference_upper=refs["upper"],
                )
                if not self.bounds.is_calibratable(metric_id):
                    self.logger.warning(
                        "%s: %s", metric_id, self.bounds.exclusion_reason(metric_id)
                    )
