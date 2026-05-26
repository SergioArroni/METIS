"""Programmatic SDK for METIS.

Provides ``Evaluator`` and ``evaluate_from_config`` for library consumers
who want to call METIS from Python code rather than the CLI.
"""

from typing import Any

import pandas as pd

from ..application.orchestrator import Orchestrator
from ..domain.entities import RunSummary


class Evaluator:
    """High-level API for running evaluations programmatically."""

    def __init__(self) -> None:
        self._orchestrator = Orchestrator()

    def run_from_config(self, config_path: str) -> RunSummary:
        """Run the full pipeline from a YAML config file."""
        return self._orchestrator.run(config_path)

    def evaluate(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
        config: dict[str, Any],
    ) -> RunSummary:
        """Evaluate pre-loaded DataFrames.

        The seed is read from config["reproducibility"]["seed"] (default: 42).
        """
        seed = config.get("reproducibility", {}).get("seed", 42)
        return self._orchestrator.evaluate_dataframes(real, synth, config, seed)


def evaluate_from_config(config_path: str) -> RunSummary:
    """Convenience function — run evaluation from a config path."""
    return Evaluator().run_from_config(config_path)
