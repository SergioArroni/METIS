"""Lower Bound Calibration Strategy (Real vs Noise).

Establishes the lower bound (worst case) by evaluating metrics between
real data and uniform random noise.
"""

import logging
from pathlib import Path

import pandas as pd

from metis.calibrate.strategies.base import BaseCalibrationStrategy


class LowerBoundStrategy(BaseCalibrationStrategy):
    """
    Calibration strategy for lower bounds (Real vs Noise).

    Generates uniform random noise and evaluates metrics against the real
    dataset to establish the worst possible scenario.
    """

    def __init__(self, evaluator=None, noise_generator=None, logger: logging.Logger = None):
        super().__init__(logger=logger)
        self._evaluator_class = evaluator
        self.noise_generator = noise_generator
        self._evaluator = None

    def get_strategy_name(self) -> str:
        return "Real vs Noise (Uniform)"

    def _iteration_seed(self, base_seed: int, index: int) -> int:
        """Same seed scheme as upper strategy and benchmark."""
        return base_seed + index

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
        import yaml

        from metis.calibrate.utils.evaluator import InMemoryEvaluator

        evaluator_cls = self._evaluator_class or InMemoryEvaluator
        self._evaluator = evaluator_cls(config_template_path)

        # Provide the schema to the noise generator so it can distinguish
        # columns like TotalCharges (str in the CSV but ``continuous`` in
        # the YAML schema) and generate numeric noise for them.
        config_path = Path(config_template_path)
        with config_path.open(encoding="utf-8") as fh:
            raw_schema = yaml.safe_load(fh).get("data", {}).get("schema", {})

        # Normalize schema: extract 'type' from dict values
        schema = {}
        for col, spec in raw_schema.items():
            if isinstance(spec, dict):
                schema[col] = spec.get("type", "categorical")
            elif isinstance(spec, str):
                schema[col] = spec
            else:
                schema[col] = "categorical"  # Fallback

        if schema and hasattr(self.noise_generator, "_schema"):
            self.noise_generator._schema = schema

        return super().calibrate(
            real_data,
            config_template_path,
            n_iterations,
            sample_size,
            base_seed,
            n_jobs,
        )

    def _execute_iteration(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        sample_size: int,
        seed: int,
    ) -> dict[str, float]:
        """Execute an iteration injecting the noise_generator."""
        return self._run_single_iteration(
            real_data,
            config_template_path,
            sample_size,
            seed,
        )

    def _run_single_iteration(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        sample_size: int,
        seed: int,
    ) -> dict[str, float]:
        """
        Run a single iteration (Real vs Noise).

        Generates uniform noise with the same size as the full real dataset
        and evaluates metrics between them.  Using the complete dataset
        (instead of a bootstrap sample) ensures that the lower-bound
        scores are directly comparable to the benchmark's uniform_noise
        generator results.
        """
        n_samples = len(real_data)

        noise_sample = self.noise_generator.generate(
            reference_data=real_data,
            n_samples=n_samples,
            seed=seed,
        )

        return self._evaluator.evaluate(
            real_data=real_data,
            synth_data=noise_sample,
            seed=seed,
        )
