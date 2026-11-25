"""Lower Bound Calibration Strategy (worst case per family).

* fidelity: uniform noise against the generator's training rows;
* privacy: an exact copy of the training rows (the most a generator can
  leak), evaluated with the holdout, as a synthetic table is.

Uniform noise used to be the privacy floor too, but noise is perfectly
private: with the relative privacy metrics it scores ≈1, like the upper
reference, and the bounds came out inverted (study ``calib_v2``).
"""

import logging
from pathlib import Path

import pandas as pd

from metis.calibrate.strategies.base import BaseCalibrationStrategy, tercile_split


class LowerBoundStrategy(BaseCalibrationStrategy):
    """Noise (fidelity) and exact copy (privacy) against the training third."""

    def __init__(self, evaluator=None, noise_generator=None, logger: logging.Logger = None):
        super().__init__(logger=logger)
        self._evaluator_class = evaluator
        self.noise_generator = noise_generator
        self._evaluator = None

    def get_strategy_name(self) -> str:
        return "Uniform noise (fidelity) / exact copy (privacy), with holdout"

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

        # The noise generator needs the schema to tell, e.g., TotalCharges
        # (str in the CSV, ``continuous`` in the YAML) from a categorical.
        with Path(config_template_path).open(encoding="utf-8") as fh:
            raw_schema = (yaml.safe_load(fh) or {}).get("data", {}).get("schema", {})
        if not isinstance(raw_schema, dict):  # "auto": the noise follows the dtypes
            raw_schema = {}
        schema = {
            col: (spec.get("type", "nominal") if isinstance(spec, dict) else spec)
            if isinstance(spec, (dict, str))
            else "nominal"
            for col, spec in raw_schema.items()
        }
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

    def _run_single_iteration(
        self,
        real_data: pd.DataFrame,
        config_template_path: str,
        sample_size: int,
        seed: int,
    ):
        """One iteration: two evaluations, merged by family."""
        train, holdout, unseen = tercile_split(real_data, sample_size, seed, self._stratify_by, self._split)
        # References at the size of the unseen part, as the benchmark's tables
        # (the holdout has that size with 60/20/20 and with thirds, not with
        # every ``calibration.split``).
        n_rows = len(unseen)
        noise = self.noise_generator.generate(
            reference_data=real_data, n_samples=n_rows, seed=seed
        )
        copy = train.sample(n=min(n_rows, len(train)), random_state=seed).reset_index(drop=True)
        parts = []
        for family, synth in (("fidelity", noise), ("privacy", copy)):
            # A config without metrics of a family has nothing to anchor there
            # (an empty metric list is a configuration error of the evaluation).
            if not self._evaluator.metrics_of((family,)):
                continue
            part = self._evaluator.evaluate(
                real_data=train, synth_data=synth, seed=seed, holdout=holdout, families=(family,)
            )
            parts.append((part, family))
        merged: list[dict] = [{}, {}, {}]
        for part, family in parts:
            for out, values in zip(merged, part, strict=True):
                if family in values:
                    out[family] = values[family]
        return tuple(merged)
