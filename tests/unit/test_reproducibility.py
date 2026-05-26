"""Tests for metis.shared.reproducibility — seed management.

WHY: Reproducibility is a core requirement for scientific evaluation.
If set_global_seed doesn't actually fix the random state, results are
non-deterministic and experiments cannot be replicated.
"""

import os
import random

import numpy as np

from metis.shared.reproducibility import (
    configure_deterministic_mode,
    get_seed_for_run,
    set_global_seed,
)


class TestSetGlobalSeed:
    """Verify that seeding produces deterministic sequences."""

    def test_numpy_deterministic(self):
        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_python_random_deterministic(self):
        set_global_seed(42)
        a = [random.random() for _ in range(5)]
        set_global_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_differ(self):
        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(99)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)

    def test_sets_hash_seed_env(self):
        set_global_seed(123)
        assert os.environ["PYTHONHASHSEED"] == "123"


class TestGetSeedForRun:
    """Arithmetic seed derivation for multi-run experiments."""

    def test_first_run(self):
        assert get_seed_for_run(42, 0) == 42

    def test_subsequent_runs(self):
        assert get_seed_for_run(42, 1) == 43
        assert get_seed_for_run(42, 5) == 47

    def test_different_base_seeds(self):
        assert get_seed_for_run(0, 3) == 3
        assert get_seed_for_run(100, 3) == 103


class TestConfigureDeterministicMode:
    def test_sets_thread_env_vars(self):
        configure_deterministic_mode(enable=True)
        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("MKL_NUM_THREADS") == "1"
        assert os.environ.get("NUMEXPR_NUM_THREADS") == "1"
