"""
Calibration system for METIS.

Provides calibration bounds estimation for metric score normalization.

API:
  - MetricCalibrator: Main orchestrator
  - CalibrationBounds: Bounds storage
  - AggregatorTuner: Aggregator optimization

Protocols (from domain.contracts):
  - CalibrationStrategy, CalibrationEvaluator, BoundsStorage, NoiseGenerator
"""

# Re-export protocols from domain for convenience
from metis.domain.contracts import (
    BoundsStorage,
    CalibrationEvaluator,
    CalibrationStrategy,
    NoiseGenerator,
)

from .core import CalibrationBounds, MetricCalibrator
from .optimization import AggregatorTuner
from .strategies import LowerBoundStrategy, UpperBoundStrategy
from .utils import InMemoryEvaluator, UniformNoiseGenerator

__all__ = [
    # Main API
    "MetricCalibrator",
    "CalibrationBounds",
    "AggregatorTuner",
    # Protocols (from domain.contracts)
    "CalibrationStrategy",
    "CalibrationEvaluator",
    "BoundsStorage",
    "NoiseGenerator",
    # Implementations
    "UpperBoundStrategy",
    "LowerBoundStrategy",
    "InMemoryEvaluator",
    "UniformNoiseGenerator",
]
