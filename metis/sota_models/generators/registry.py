"""
Generator registry — Strategy + Registry pattern for generator orchestration.

The ``GeneratorRegistry`` acts as a centralised catalogue of available
``BaseGenerator`` strategies.  New generators are registered by key,
and instances are created via ``create()`` without the caller needing
to know the concrete class.

Usage::

    from metis.sota_models.generators.registry import GeneratorRegistry

    # Register a custom generator
    GeneratorRegistry.register("my_gen", MyCustomGenerator)

    # Create an instance by name (Strategy selection)
    gen = GeneratorRegistry.create("ctgan", epochs=100, random_state=42)

    # List all available keys
    print(GeneratorRegistry.available())
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator


class GeneratorRegistry:
    """
    Central registry that maps string keys to ``BaseGenerator`` subclasses.

    Combines:
    * **Strategy pattern** – every registered class is a concrete strategy
      implementing the ``BaseGenerator`` interface (fit / generate).
    * **Registry pattern** – callers request a strategy by name and receive
      an instance without coupling to the concrete class.
    """

    _registry: dict[str, type[BaseGenerator]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, key: str, generator_class: type[BaseGenerator]) -> None:
        """
        Register a generator class under *key* (case-insensitive).

        Raises:
            TypeError: If *generator_class* is not a ``BaseGenerator`` subclass.
        """
        if not (isinstance(generator_class, type) and issubclass(generator_class, BaseGenerator)):
            raise TypeError(
                f"Only BaseGenerator subclasses can be registered, got {generator_class!r}"
            )
        cls._registry[key.lower()] = generator_class

    @classmethod
    def unregister(cls, key: str) -> None:
        """Remove a previously registered key (no-op if absent)."""
        cls._registry.pop(key.lower(), None)

    # ------------------------------------------------------------------
    # Factory / Strategy selection
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, key: str, **kwargs: Any) -> BaseGenerator:
        """
        Instantiate the generator registered under *key*.

        Args:
            key: Case-insensitive registry key (e.g. ``"ctgan"``).
            **kwargs: Forwarded to the generator constructor.

        Returns:
            A ready-to-use ``BaseGenerator`` instance.

        Raises:
            ValueError: If *key* is not registered.
        """
        generator_class = cls._registry.get(key.lower())
        if generator_class is None:
            raise ValueError(
                f"Unknown generator: {key}. Available: {', '.join(sorted(cls._registry))}"
            )
        return generator_class(**kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def available(cls) -> list[str]:
        """Return sorted list of registered keys."""
        return sorted(cls._registry)

    @classmethod
    def get_class(cls, key: str) -> type[BaseGenerator] | None:
        """Return the class registered under *key*, or ``None``."""
        return cls._registry.get(key.lower())

    @classmethod
    def as_dict(cls) -> dict[str, type[BaseGenerator]]:
        """Return a copy of the full registry mapping."""
        return dict(cls._registry)


# ======================================================================
# Default registrations — executed on first import of this module.
# ======================================================================


def _register_defaults() -> None:
    """Populate the registry with all built-in generators."""
    from .adsgan import ADSGANGenerator
    from .bayesian_network import BayesianNetworkGenerator
    from .bootstrap import RandomSamplingGenerator
    from .cart import CARTGenerator
    from .ctgan import CTGANGenerator
    from .delete_impute import DeleteImputeMeanGenerator, DeleteImputeZeroGenerator
    from .dpctgan import DPCTGANGenerator
    from .gaussian_copula import GaussianCopulaGenerator
    from .real_data import RealDataGenerator
    from .smotenc import SMOTENCGenerator
    from .tvae import TVAEGenerator
    from .uniform_noise import UniformNoiseGenerator

    # Baselines
    GeneratorRegistry.register("smotenc", SMOTENCGenerator)
    GeneratorRegistry.register("bootstrap", RandomSamplingGenerator)
    GeneratorRegistry.register("delete_zero", DeleteImputeZeroGenerator)
    GeneratorRegistry.register("delete_mean", DeleteImputeMeanGenerator)
    GeneratorRegistry.register("uniform_noise", UniformNoiseGenerator)
    GeneratorRegistry.register("real_data", RealDataGenerator)
    GeneratorRegistry.register("gaussian_copula", GaussianCopulaGenerator)

    # SOTA models
    GeneratorRegistry.register("bn", BayesianNetworkGenerator)
    GeneratorRegistry.register("bayesian_network", BayesianNetworkGenerator)
    GeneratorRegistry.register("cart", CARTGenerator)
    GeneratorRegistry.register("ctgan", CTGANGenerator)
    GeneratorRegistry.register("adsgan", ADSGANGenerator)
    GeneratorRegistry.register("tvae", TVAEGenerator)
    GeneratorRegistry.register("dpctgan", DPCTGANGenerator)


_register_defaults()
