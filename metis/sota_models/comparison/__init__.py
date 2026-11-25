"""Comparison package."""

from .orchestrator import BenchmarkOrchestrator
from .reporter import BenchmarkReporter, compare_across_datasets
from .statistical_tests import (
    FriedmanNemenyiTest,
    StatisticalTest,
    WilcoxonTest,
    get_statistical_test,
)

__all__ = [
    "BenchmarkOrchestrator",
    "BenchmarkReporter",
    "compare_across_datasets",
    "StatisticalTest",
    "FriedmanNemenyiTest",
    "WilcoxonTest",
    "get_statistical_test",
]
