"""
Base generator interface for synthetic data generation methods.

This module defines the abstract base class that all generators (baseline and SOTA)
must implement to ensure consistent interface across benchmarking.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseGenerator(ABC):
    """
    Abstract base class for synthetic data generators.

    All generators (baseline and SOTA models) must inherit from this class
    and implement the generate method. This ensures a consistent interface
    for the benchmarking orchestrator.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the generator.

        Args:
            name: Human-readable name for the generator
            **kwargs: Additional generator-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        real_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        """
        Fit the generator to the real data.

        Args:
            real_data: Real dataset to learn from
            categorical_columns: list of categorical column names
            ordinal_columns: dict mapping ordinal column names to their ordered levels
            continuous_columns: list of continuous column names
        """
        pass

    @abstractmethod
    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data.

        Args:
            n_samples: Number of synthetic samples to generate

        Returns:
            DataFrame with synthetic data matching the schema of the real data

        Raises:
            RuntimeError: If generate is called before fit
        """
        pass

    def fit_generate(
        self,
        real_data: pd.DataFrame,
        n_samples: int,
        categorical_columns: list[str] | None = None,
        ordinal_columns: dict[str, list] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Convenience method to fit and generate in one call.

        Args:
            real_data: Real dataset to learn from
            n_samples: Number of synthetic samples to generate
            categorical_columns: list of categorical column names
            ordinal_columns: dict mapping ordinal column names to their ordered levels
            continuous_columns: list of continuous column names

        Returns:
            DataFrame with synthetic data
        """
        self.fit(
            real_data=real_data,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            continuous_columns=continuous_columns,
        )
        return self.generate(n_samples=n_samples)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
