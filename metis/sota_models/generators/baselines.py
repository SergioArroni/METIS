"""
Backward-compatibility shim -- imports moved to individual modules.

This file re-exports all baseline generators so that existing code using
``from metis.sota_models.generators.baselines import ...`` continues to work.
"""

from .bootstrap import RandomSamplingGenerator
from .delete_impute import (
    DeleteImputeGenerator,
    DeleteImputeMeanGenerator,
    DeleteImputeZeroGenerator,
)
from .gaussian_copula import GaussianCopulaGenerator
from .real_data import RealDataGenerator
from .smotenc import SMOTENCGenerator
from .uniform_noise import UniformNoiseGenerator

__all__ = [
    "SMOTENCGenerator",
    "RandomSamplingGenerator",
    "DeleteImputeGenerator",
    "DeleteImputeZeroGenerator",
    "DeleteImputeMeanGenerator",
    "UniformNoiseGenerator",
    "RealDataGenerator",
    "GaussianCopulaGenerator",
]
