"""Comparison package."""

from .orchestrator import BenchmarkOrchestrator
from .reporter import BenchmarkReporter
from .statistical_tests import (
    FriedmanNemenyiTest,
    StatisticalTest,
    WilcoxonTest,
    get_statistical_test,
)

__all__ = [
    "BenchmarkOrchestrator",
    "BenchmarkReporter",
    "StatisticalTest",
    "FriedmanNemenyiTest",
    "WilcoxonTest",
    "get_statistical_test",
]
