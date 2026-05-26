"""
SOTA Models - Benchmark framework for comparing synthetic data generators.

This package provides tools for benchmarking multiple synthetic data generation
methods (baselines and state-of-the-art) using METIS evaluation metrics.
"""

__version__ = "0.1.0"

from .comparison.orchestrator import BenchmarkOrchestrator
from .comparison.reporter import BenchmarkReporter
from .comparison.statistical_tests import (
    FriedmanNemenyiTest,
    StatisticalTest,
    WilcoxonTest,
    get_statistical_test,
)
from .generators.adsgan import ADSGANGenerator
from .generators.base import BaseGenerator
from .generators.bayesian_network import BayesianNetworkGenerator
from .generators.bootstrap import RandomSamplingGenerator
from .generators.cart import CARTGenerator
from .generators.ctgan import CTGANGenerator
from .generators.delete_impute import (
    DeleteImputeGenerator,
    DeleteImputeMeanGenerator,
    DeleteImputeZeroGenerator,
)
from .generators.gaussian_copula import GaussianCopulaGenerator
from .generators.registry import GeneratorRegistry
from .generators.smotenc import SMOTENCGenerator

__all__ = [
    # Version
    "__version__",
    # Base
    "BaseGenerator",
    "GeneratorRegistry",
    # Baseline generators
    "SMOTENCGenerator",
    "RandomSamplingGenerator",
    "DeleteImputeGenerator",
    "DeleteImputeZeroGenerator",
    "DeleteImputeMeanGenerator",
    "GaussianCopulaGenerator",
    # SOTA generators
    "BayesianNetworkGenerator",
    "CARTGenerator",
    "CTGANGenerator",
    "ADSGANGenerator",
    # Orchestration
    "BenchmarkOrchestrator",
    "BenchmarkReporter",
    # Statistical tests
    "StatisticalTest",
    "FriedmanNemenyiTest",
    "WilcoxonTest",
    "get_statistical_test",
]
