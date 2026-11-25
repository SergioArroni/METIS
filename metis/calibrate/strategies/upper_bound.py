"""Upper Bound Calibration Strategy (unseen real rows).

Establishes the best case of every calibrated metric: a sample of real rows
the generator never saw, evaluated against the generator's training rows with
the holdout given, exactly as a synthetic table is evaluated.
"""

import logging

import pandas as pd

from metis.calibrate.core.bounds import CALIBRATED_FAMILIES
from metis.calibrate.strategies.base import BaseCalibrationStrategy, tercile_split


class UpperBoundStrategy(BaseCalibrationStrategy):
    """Unseen real third vs training third, holdout third given (fidelity and privacy)."""

    def __init__(self, evaluator=None, logger: logging.Logger = None):
        super().__init__(logger=logger)
        self._evaluator_class = evaluator
        self._evaluator = None

    def get_strategy_name(self) -> str:
        return "Unseen real vs training real (thirds, with holdout)"

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
    ):
        train, holdout, unseen = tercile_split(real_data, sample_size, seed, self._stratify_by, self._split)
        return self._evaluator.evaluate(
            real_data=train,
            synth_data=unseen,
            seed=seed,
            holdout=holdout,
            families=CALIBRATED_FAMILIES,
        )
