"""Base class for calibration strategies.

Centralizes common logic: sequential iteration execution,
result merging, and summary logging.

Uses the Template Method pattern: the calibration flow is fixed
(logging, sequential execution, merge, summary), but each subclass
defines the logic for a single iteration.
"""

import logging
import time

import numpy as np
import pandas as pd


class BaseCalibrationStrategy:
    """
    Base implementation for calibration strategies.

    Conforms to the CalibrationStrategy Protocol defined in
    metis.domain.contracts.

    Template Method hooks for subclasses:
    - _run_single_iteration: per-iteration logic (required)
    - _iteration_seed: how to compute per-iteration seed (optional)
    - get_strategy_name: descriptive name (required)
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        # Per-metric normalized values collected during calibrate(), keyed by family.
        # Structure: {family: [{metric_id: value, ...}, ...]}
        self._metric_values: dict[str, list[dict[str, float]]] = {}
        # Per-metric raw (pre-normalization) values, same structure.
        self._raw_metric_values: dict[str, list[dict[str, float]]] = {}

    @property
    def metric_values_per_iteration(self) -> dict[str, list[dict[str, float]]]:
        """Per-family, per-iteration normalized metric values.

        Available after ``calibrate()`` completes.  Used by
        ``MetricCalibrator`` for aggregator tuning: different aggregation
        functions are applied to these values to find the one that best
        approximates theoretical bounds.

        Structure: ``{family: [{metric_id: value, ...}, ...]}``.  Each
        dict represents one calibration iteration.
        """
        return self._metric_values

    @property
    def raw_metric_values_per_iteration(self) -> dict[str, list[dict[str, float]]]:
        """Per-family, per-iteration raw (pre-normalization) metric values.

        Same structure as :attr:`metric_values_per_iteration` but with
        the original metric outputs before normalization to [0, 1].
        """
        return self._raw_metric_values

    def calibrate(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        n_iterations: int,
        sample_size: int,
        base_seed: int,
        n_jobs: int = 1,
    ) -> dict[str, list[float]]:
        """Execute calibration sequentially and merge results.

        Args:
            real_data: Complete real dataset.
            config_template_path: Path to YAML config template.
            n_iterations: Number of calibration iterations.
            sample_size: Sample size per iteration.
            base_seed: Base seed for reproducibility.
            n_jobs: Ignored (kept for interface compatibility).

        Returns:
            Dictionary mapping family -> list of scores per iteration.
        """
        strategy_name = self.get_strategy_name()
        self.logger.info("=" * 70)
        self.logger.info("CALIBRATION: %s", strategy_name)
        self.logger.info("=" * 70)
        self.logger.info("Iterations: %d", n_iterations)
        self.logger.info("Sample size: %d", sample_size)

        start_time = time.time()
        iteration_times: list[float] = []
        results_by_family: dict[str, list[float]] = {}
        self._metric_values = {}  # reset
        self._raw_metric_values = {}  # reset

        for i in range(n_iterations):
            seed = self._iteration_seed(base_seed, i)
            iter_start = time.time()
            try:
                result = self._execute_iteration(real_data, config_template_path, sample_size, seed)
                iter_time = time.time() - iter_start
                iteration_times.append(iter_time)

                # Evaluator contract returns (family_scores, metric_values, raw_values).
                if not (isinstance(result, tuple) and len(result) == 3):
                    raise TypeError(
                        "CalibrationEvaluator.evaluate must return a 3-tuple "
                        "(family_scores, metric_values, raw_values); "
                        f"got {type(result).__name__} with len="
                        f"{len(result) if isinstance(result, tuple) else 'n/a'}"
                    )
                family_scores, metric_values, raw_values = result

                for family, score in family_scores.items():
                    results_by_family.setdefault(family, []).append(score)

                # Accumulate per-metric normalized values for aggregator tuning
                for family, values in metric_values.items():
                    self._metric_values.setdefault(family, []).append(values)

                # Accumulate per-metric raw values for traceability
                for family, values in raw_values.items():
                    self._raw_metric_values.setdefault(family, []).append(values)

                self.logger.info(
                    "  Iteration %d/%d (%.1fs): %s",
                    i + 1,
                    n_iterations,
                    iter_time,
                    " | ".join(f"{k}={v:.4f}" for k, v in family_scores.items()),
                )
            except Exception as e:
                self.logger.error("  Iteration %d failed: %s", i + 1, e)

        total_time = time.time() - start_time
        self._log_summary(strategy_name, results_by_family, iteration_times, total_time)
        return results_by_family

    def _execute_iteration(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        sample_size: int,
        seed: int,
    ) -> dict[str, float]:
        """Execute a single iteration. Subclasses may override to inject
        additional dependencies (e.g. noise_generator)."""
        return self._run_single_iteration(real_data, config_template_path, sample_size, seed)

    def _iteration_seed(self, base_seed: int, index: int) -> int:
        """Compute the seed for a given iteration. Subclasses may override."""
        return base_seed + index

    def _run_single_iteration(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        sample_size: int,
        seed: int,
    ) -> dict[str, float]:
        """Per-iteration logic. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_strategy_name(self) -> str:
        """Descriptive name for the strategy. Must be implemented by subclasses."""
        raise NotImplementedError

    def _log_summary(
        self,
        name: str,
        results: dict[str, list[float]],
        times: list[float],
        total: float,
    ) -> None:
        """Log calibration summary."""
        self.logger.info("\n" + "-" * 70)
        self.logger.info("%s Summary:", name)
        self.logger.info("-" * 70)
        self.logger.info("  Total time: %.1fs (%.1fm)", total, total / 60)
        if times:
            self.logger.info("  Avg time per iteration: %.1fs", np.mean(times))
            self.logger.info(
                "  Min/Max iteration time: %.1fs / %.1fs",
                np.min(times),
                np.max(times),
            )
        for family in sorted(results.keys()):
            scores = results[family]
            self.logger.info(
                "  %8s: mean=%.4f std=%.4f [%.4f, %.4f]",
                family,
                np.mean(scores),
                np.std(scores),
                np.min(scores),
                np.max(scores),
            )
