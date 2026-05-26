"""Main calibrator - Strategy orchestrator.

Coordinates upper/lower bound calibration strategies, computes final bounds,
and manages the calibration flow.

Architecture (correct order):
    1. Run upper (Real-vs-Real) and lower (Real-vs-Noise) strategies.
       Each iteration produces per-metric raw values AND default-aggregated
       family scores.
    2. Tune per-family aggregators using the per-metric raw data.
       The tuner tries every candidate function and picks the one whose
       aggregated scores are closest to the theoretical limits [0, 1].
    3. Re-aggregate the per-metric data with the **selected** aggregators
       to obtain N corrected family scores per strategy.
    4. Compute min/max bounds from the re-aggregated scores.
    5. Store optimal aggregators inside ``CalibrationBounds`` so the
       benchmark evaluation uses the same functions.

Uses Strategy Pattern to delegate to specialized strategies
and Dependency Injection to decouple from concrete implementations.
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from metis.calibrate.core.bounds import CalibrationBounds


class MetricCalibrator:
    """
    Calibration orchestrator for metric bounds estimation.

    Coordinates upper/lower bound strategies and computes the final
    bounds for family-level score normalization.
    """

    def __init__(
        self,
        upper_strategy=None,
        lower_strategy=None,
        bounds_storage=None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

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
        n_iterations: int = 5,
        sample_size: int = 1000,
        base_seed: int = 42,
        n_jobs: int = -1,
        tune_aggregators: bool = True,
    ) -> CalibrationBounds:
        """
        Execute full calibration using injected strategies.

        Flow:
        1. Run upper bound strategy  (collect family scores + per-metric data)
        2. Run lower bound strategy  (collect family scores + per-metric data)
        3. Tune per-family aggregators from per-metric data
        4. Re-aggregate per-metric data with selected aggregators
        5. Compute final bounds (min/max) from re-aggregated scores
        """
        sample_percentage = (sample_size / len(real_data) * 100.0) if sample_size else 100.0
        dataset_size_mb = real_data.memory_usage(deep=True).sum() / (1024**2)

        self.logger.info("=" * 70)
        self.logger.info("METIS CALIBRATION SYSTEM")
        self.logger.info("=" * 70)
        self.logger.info("Upper strategy: %s", self.upper_strategy.get_strategy_name())
        self.logger.info("Lower strategy: %s", self.lower_strategy.get_strategy_name())
        self.logger.info(
            "Dataset: %d rows, %d columns (%.1fMB)",
            len(real_data),
            len(real_data.columns),
            dataset_size_mb,
        )
        self.logger.info("Sample: %d rows (%.1f%%)", sample_size, sample_percentage)
        self.logger.info("Iterations: %d per strategy", n_iterations)
        self.logger.info("Tune aggregators: %s", tune_aggregators)

        calibration_start_time = time.time()
        upper_results: dict[str, list[float]] = {}
        lower_results: dict[str, list[float]] = {}
        upper_time = 0.0
        lower_time = 0.0

        # -------------------------------------------------------
        # Step 1: Upper bounds (Real-vs-Real split-half)
        # -------------------------------------------------------
        try:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("STEP 1: UPPER BOUNDS")
            self.logger.info("=" * 70)

            upper_start = time.time()
            upper_results = self.upper_strategy.calibrate(
                real_data=real_data,
                config_template_path=config_template_path,
                n_iterations=n_iterations,
                sample_size=sample_size,
                base_seed=base_seed,
                n_jobs=n_jobs,
            )
            upper_time = time.time() - upper_start
            self.logger.info("Upper bounds completed in %.1fs (%.1fm)", upper_time, upper_time / 60)
        except Exception as e:
            self.logger.exception("Error in upper bounds: %s", e)
            self.logger.warning("Continuing with lower bounds (upper will be empty)")

        # -------------------------------------------------------
        # Step 2: Lower bounds (Real-vs-Noise)
        # -------------------------------------------------------
        try:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("STEP 2: LOWER BOUNDS")
            self.logger.info("=" * 70)

            lower_start = time.time()
            lower_results = self.lower_strategy.calibrate(
                real_data=real_data,
                config_template_path=config_template_path,
                n_iterations=n_iterations,
                sample_size=sample_size,
                base_seed=base_seed,
                n_jobs=n_jobs,
            )
            lower_time = time.time() - lower_start
            self.logger.info("Lower bounds completed in %.1fs (%.1fm)", lower_time, lower_time / 60)
        except Exception as e:
            self.logger.exception("Error in lower bounds: %s", e)
            self.logger.warning("Continuing with partial results")

        if not upper_results and not lower_results:
            raise RuntimeError("Both strategies failed. No results to save.")

        # -------------------------------------------------------
        # Step 3: Tune aggregators + re-aggregate + compute bounds
        # -------------------------------------------------------
        upper_metric_data = self.upper_strategy.metric_values_per_iteration
        lower_metric_data = self.lower_strategy.metric_values_per_iteration

        has_metric_data = bool(upper_metric_data) and bool(lower_metric_data)

        if tune_aggregators and has_metric_data:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("STEP 3: TUNE AGGREGATORS (per-metric re-aggregation)")
            self.logger.info("=" * 70)

            from metis.calibrate.optimization.aggregator_tuner import AggregatorTuner

            tuner = AggregatorTuner(logger=self.logger)
            tuning_results = tuner.tune_from_metrics(upper_metric_data, lower_metric_data)
            optimal = tuning_results.get("optimal", {})

            # Store full per-family + composite configuration
            self.bounds.optimal_aggregators = optimal
            self.logger.info("Optimal aggregators: %s", optimal)

            # Re-aggregate per-metric data with the tuned per-family aggregators
            self.logger.info("\n" + "=" * 70)
            self.logger.info("STEP 4: RE-AGGREGATE WITH TUNED AGGREGATORS")
            self.logger.info("=" * 70)

            upper_results = AggregatorTuner.reaggregate(upper_metric_data, optimal)
            lower_results = AggregatorTuner.reaggregate(lower_metric_data, optimal)

            for family in sorted(upper_results):
                u_scores = upper_results[family]
                l_scores = lower_results.get(family, [])
                self.logger.info(
                    "  %s: upper=[%.4f..%.4f] lower=[%.4f..%.4f]",
                    family,
                    min(u_scores) if u_scores else 0,
                    max(u_scores) if u_scores else 0,
                    min(l_scores) if l_scores else 0,
                    max(l_scores) if l_scores else 0,
                )
        else:
            if tune_aggregators and not has_metric_data:
                self.logger.warning(
                    "Per-metric data unavailable -- falling back to default aggregators. "
                    "This normally means the strategies did not collect metric values."
                )
            # Use the default-aggregated family scores as-is
            self.logger.info("\n" + "=" * 70)
            self.logger.info("STEP 3: COMPUTING BOUNDS (default aggregators)")
            self.logger.info("=" * 70)

        # -------------------------------------------------------
        # Step 5: Compute min/max bounds from (re-aggregated) scores
        # -------------------------------------------------------
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 5: COMPUTING FINAL BOUNDS (min/max)")
        self.logger.info("=" * 70)

        self._compute_bounds(upper_results, lower_results)

        # -------------------------------------------------------
        # Metadata
        # -------------------------------------------------------
        total_calibration_time = time.time() - calibration_start_time
        self.bounds.set_metadata("n_iterations", n_iterations)
        self.bounds.set_metadata("sample_size", sample_size)
        self.bounds.set_metadata("sample_percentage", sample_percentage)
        self.bounds.set_metadata("base_seed", base_seed)
        self.bounds.set_metadata("upper_strategy", self.upper_strategy.get_strategy_name())
        self.bounds.set_metadata("lower_strategy", self.lower_strategy.get_strategy_name())
        self.bounds.set_metadata("calibration_time_seconds", total_calibration_time)
        self.bounds.set_metadata("upper_time_seconds", upper_time)
        self.bounds.set_metadata("lower_time_seconds", lower_time)
        self.bounds.set_metadata("tune_aggregators", tune_aggregators)
        self.bounds.set_metadata("dataset_rows", len(real_data))
        self.bounds.set_metadata("dataset_columns", len(real_data.columns))

        # -------------------------------------------------------
        # Store per-metric values for full traceability
        # -------------------------------------------------------
        if has_metric_data:
            upper_raw_data = self.upper_strategy.raw_metric_values_per_iteration
            lower_raw_data = self.lower_strategy.raw_metric_values_per_iteration

            metric_details: dict[str, Any] = {}
            for family in sorted(
                set(list(upper_metric_data.keys()) + list(lower_metric_data.keys()))
            ):
                upper_iters = upper_metric_data.get(family, [])
                lower_iters = lower_metric_data.get(family, [])
                upper_raw_iters = upper_raw_data.get(family, [])
                lower_raw_iters = lower_raw_data.get(family, [])
                metric_details[family] = {
                    "upper_iterations": [
                        {
                            "iteration": i + 1,
                            **{
                                mid: {
                                    "normalized": nval,
                                    "raw": upper_raw_iters[i].get(mid, nval)
                                    if i < len(upper_raw_iters)
                                    else nval,
                                }
                                for mid, nval in vals.items()
                            },
                        }
                        for i, vals in enumerate(upper_iters)
                    ],
                    "lower_iterations": [
                        {
                            "iteration": i + 1,
                            **{
                                mid: {
                                    "normalized": nval,
                                    "raw": lower_raw_iters[i].get(mid, nval)
                                    if i < len(lower_raw_iters)
                                    else nval,
                                }
                                for mid, nval in vals.items()
                            },
                        }
                        for i, vals in enumerate(lower_iters)
                    ],
                }
            self.bounds.metric_details = metric_details

        self.logger.info("\n" + "=" * 70)
        self.logger.info("CALIBRATION COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(
            "TOTAL TIME: %.1fs (%.1fm)", total_calibration_time, total_calibration_time / 60
        )
        if total_calibration_time > 0:
            self.logger.info(
                "   - Upper bounds: %.1fs (%.1f%%)",
                upper_time,
                upper_time / total_calibration_time * 100,
            )
            self.logger.info(
                "   - Lower bounds: %.1fs (%.1f%%)",
                lower_time,
                lower_time / total_calibration_time * 100,
            )
        self.logger.info(self.bounds.get_summary())

        return self.bounds

    def _compute_bounds(
        self,
        upper_results: dict[str, list[float]],
        lower_results: dict[str, list[float]],
    ) -> None:
        """
        Compute final bounds with automatic inversion detection.

        Uses min/max of calibration iterations:
        - Upper bound = min(best-strategy iterations)  → ceiling
        - Lower bound = max(worst-strategy iterations) → floor

        This guarantees that the reference generators (real_data, uniform_noise)
        always calibrate to exactly 0.0 and 1.0:
        - Any score ≥ min(best) → calibrated ≥ 1.0 → clipped to 1.0
        - Any score ≤ max(worst) → calibrated ≤ 0.0 → clipped to 0.0

        Detects semantic inversion (e.g. privacy metrics) and swaps automatically.
        """
        self.logger.info("Computing final bounds from iteration results (min/max)...")
        self.logger.info(
            "  Upper families: %s", sorted(upper_results.keys()) if upper_results else "(empty)"
        )
        self.logger.info(
            "  Lower families: %s", sorted(lower_results.keys()) if lower_results else "(empty)"
        )

        all_families = set(upper_results.keys()) | set(lower_results.keys())

        if not all_families:
            self.logger.error("No families to process (empty results)")
            return

        for family in sorted(all_families):
            if family not in upper_results or family not in lower_results:
                self.logger.warning("%s: Missing results in one strategy, skipping", family)
                continue

            try:
                upper_iterations = upper_results[family]
                lower_iterations = lower_results[family]

                self.logger.info(
                    "  %s: upper has %d iterations, lower has %d iterations",
                    family,
                    len(upper_iterations),
                    len(lower_iterations),
                )

                upper_min = float(np.min(upper_iterations))
                lower_max = float(np.max(lower_iterations))
                upper_median = float(np.median(upper_iterations))
                lower_median = float(np.median(lower_iterations))

                self.logger.info(
                    "  %s: median=[%.4f, %.4f] -> min/max=[%.4f, %.4f]",
                    family,
                    lower_median,
                    upper_median,
                    lower_max,
                    upper_min,
                )

                # Detect inversion (privacy metrics have inverted semantics).
                # Previously the decision used the tail overlap
                # ``upper_min < lower_max``, which triggers spurious flips when
                # noisy iterations overlap in their extremes even though the
                # medians clearly satisfy upper > lower. Use the medians as
                # the orientation signal; the tails are still used to set the
                # actual bound endpoints below.
                is_inverted = upper_median < lower_median

                # Determine if the normalization needs to flip the result.
                # For "naturally inverted" families like privacy, the upper
                # strategy (real-vs-real) is *expected* to produce lower scores
                # than noise. The bound swap alone corrects this — no flip.
                # For fidelity/utility, inversion is *unexpected* (caused by
                # the aggregator, e.g. SSD) — the swap re-orders the bounds
                # but normalization still maps high raw → 1.0, which is wrong
                # when the aggregator reverses the score ordering.
                NATURALLY_INVERTED_FAMILIES = {"privacy"}
                needs_normalization_flip = is_inverted and family not in NATURALLY_INVERTED_FAMILIES

                if is_inverted:
                    self.logger.info(
                        "%s: Inverted semantics detected (upper_min=%.4f < lower_max=%.4f)",
                        family,
                        upper_min,
                        lower_max,
                    )
                    self.logger.info("  -> Swapping iteration labels for semantic correctness")

                    # After inversion: lower_iterations are the "best" strategy
                    # upper_iterations are the "worst" strategy
                    final_lower_bound = float(np.max(upper_iterations))
                    final_upper_bound = float(np.min(lower_iterations))
                    final_lower_iterations = upper_iterations
                    final_upper_iterations = lower_iterations
                else:
                    # upper_iterations are the "best" strategy
                    # lower_iterations are the "worst" strategy
                    final_lower_bound = lower_max
                    final_upper_bound = upper_min
                    final_lower_iterations = lower_iterations
                    final_upper_iterations = upper_iterations

                self.bounds.set_bounds(
                    family=family,
                    lower_bound=final_lower_bound,
                    upper_bound=final_upper_bound,
                    lower_iterations=final_lower_iterations,
                    upper_iterations=final_upper_iterations,
                    inverted=needs_normalization_flip,
                )

                self.logger.info(
                    "  %s: [%.4f, %.4f] (inverted=%s, method=min/max)",
                    family,
                    final_lower_bound,
                    final_upper_bound,
                    is_inverted,
                )
            except Exception as e:
                self.logger.error("%s: Failed to compute bounds - %s", family, e)
                continue
