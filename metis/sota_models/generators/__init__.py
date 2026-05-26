"""
Generators package.

Each generator lives in its own module and implements the ``BaseGenerator``
strategy interface.  The ``GeneratorRegistry`` provides a centralised
Strategy + Registry pattern for discovering and instantiating generators
by name.
"""

from .adsgan import ADSGANGenerator
from .base import BaseGenerator
from .bayesian_network import BayesianNetworkGenerator
from .bootstrap import RandomSamplingGenerator
from .cart import CARTGenerator
from .ctgan import CTGANGenerator
from .delete_impute import (
    DeleteImputeGenerator,
    DeleteImputeMeanGenerator,
    DeleteImputeZeroGenerator,
)
from .dpctgan import DPCTGANGenerator
from .gaussian_copula import GaussianCopulaGenerator
from .real_data import RealDataGenerator
from .registry import GeneratorRegistry
from .smotenc import SMOTENCGenerator
from .tvae import TVAEGenerator
from .uniform_noise import UniformNoiseGenerator

__all__ = [
    # Strategy base
    "BaseGenerator",
    # Registry / orchestration
    "GeneratorRegistry",
    # Baselines
    "SMOTENCGenerator",
    "RandomSamplingGenerator",
    "DeleteImputeGenerator",
    "DeleteImputeZeroGenerator",
    "DeleteImputeMeanGenerator",
    "UniformNoiseGenerator",
    "RealDataGenerator",
    "GaussianCopulaGenerator",
    # SOTA generators
    "BayesianNetworkGenerator",
    "CARTGenerator",
    "CTGANGenerator",
    "ADSGANGenerator",
    "TVAEGenerator",
    "DPCTGANGenerator",
]
