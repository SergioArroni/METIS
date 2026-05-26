"""
Reproducibility utilities for fixing random seeds across all libraries.

This module provides functions to ensure reproducible results by setting
random seeds for all commonly used randomness sources in the evaluation pipeline.
"""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    set random seed for all randomness sources to ensure reproducibility.

    Sets seeds for:
    - Python's built-in random module
    - NumPy
    - Environment variable for hash randomization

    Args:
        seed: Integer seed value (typically 0-2^32-1)

    Example:
        >>> set_global_seed(42)
        >>> # All subsequent random operations will be reproducible
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Try to set torch seed if available (optional dependency)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Try to set tensorflow seed if available (optional dependency)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_seed_for_run(base_seed: int, run_index: int) -> int:
    """
    Calculate seed for a specific run in a multi-run experiment.

    Args:
        base_seed: Base seed value
        run_index: Zero-based index of the current run

    Returns:
        Seed value for this specific run (base_seed + run_index)

    Example:
        >>> get_seed_for_run(42, 0)  # First run
        42
        >>> get_seed_for_run(42, 1)  # Second run
        43
        >>> get_seed_for_run(42, 2)  # Third run
        44
    """
    return base_seed + run_index


def configure_deterministic_mode(enable: bool = True) -> None:
    """
    Configure deterministic behavior for scientific computing libraries.

    Note: This may impact performance but ensures reproducibility.

    Args:
        enable: If True, enable deterministic mode; if False, allow non-deterministic
    """
    # NumPy uses deterministic algorithms by default
    # PyTorch and TensorFlow settings are handled in set_global_seed()

    if enable:
        # Ensure single-threaded execution for NumPy operations
        # (can be overridden by user if needed)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
