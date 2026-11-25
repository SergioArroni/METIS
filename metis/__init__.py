"""METIS: A modular framework for synthetic tabular data evaluation."""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0.dev0"

__all__ = ["Evaluator", "evaluate_from_config"]


def __getattr__(name: str) -> Any:
    """Import the SDK on first use.

    ``import metis.domain`` stays light, and a missing dependency surfaces as
    the real ``ImportError`` (it used to be swallowed at import time, so
    ``from metis import Evaluator`` only said the name did not exist).
    """
    if name in __all__:
        from .interface import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module 'metis' has no attribute {name!r}")
