"""Upper Bound Calibration Strategy (Real vs Real).

Establishes the upper bound (best case) by evaluating metrics between
two disjoint halves of the real dataset (split-half without replacement).
"""

import logging

import numpy as np
import pandas as pd

from metis.calibrate.strategies.base import BaseCalibrationStrategy


class UpperBoundStrategy(BaseCalibrationStrategy):
    """
    Calibration strategy for upper bounds (Real vs Real).

    Splits the real dataset into two disjoint halves (split-half without
    replacement) and evaluates metrics between them to establish the best
    possible scenario.
    """

    def __init__(self, evaluator=None, logger: logging.Logger = None):
        super().__init__(logger=logger)
        self._evaluator_class = evaluator
        self._evaluator = None

    def get_strategy_name(self) -> str:
        return "Real vs Real (Split-Half)"

    def calibrate(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        n_iterations: int,
        sample_size: int,
        base_seed: int,
        n_jobs: int = 1,
    ) -> dict[str, list[float]]:
        """Create evaluator once before running iterations."""
        from metis.calibrate.utils.evaluator import InMemoryEvaluator

        evaluator_cls = self._evaluator_class or InMemoryEvaluator
        self._evaluator = evaluator_cls(config_template_path)
        return super().calibrate(
            real_data,
            config_template_path,
            n_iterations,
            sample_size,
            base_seed,
            n_jobs,
        )

    def _run_single_iteration(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        sample_size: int,
        seed: int,
    ) -> dict[str, float]:
        """
        Run a single iteration using split-half.

        Splits the real dataset into two disjoint halves (without replacement)
        and evaluates metrics between them.
        """
        rng = np.random.RandomState(seed)

        # Split-half: shuffle and split into two disjoint halves
        n = len(real_data)
        indices = rng.permutation(n)
        half = n // 2

        half_size = min(sample_size, half)

        idx_a = indices[:half_size]
        idx_b = indices[half : half + half_size]

        real_sample_1 = real_data.iloc[idx_a].reset_index(drop=True)
        real_sample_2 = real_data.iloc[idx_b].reset_index(drop=True)

        return self._evaluator.evaluate(
            real_data=real_sample_1,
            synth_data=real_sample_2,
            seed=seed,
        )
