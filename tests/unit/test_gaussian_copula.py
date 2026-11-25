"""Unit tests for GaussianCopulaGenerator."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sdv", reason="sdv optional dependency not installed")


@pytest.fixture
def generator_cls():
    from metis.sota_models.generators.gaussian_copula import GaussianCopulaGenerator

    return GaussianCopulaGenerator


@pytest.fixture
def sample_data():
    np.random.seed(0)
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 65, size=100),
            "salary": np.random.uniform(20000, 80000, size=100).round(2),
            "department": np.random.choice(["HR", "IT", "Sales"], size=100),
        }
    )


class TestGaussianCopulaGenerator:
    def test_fit_generate_contract(self, sample_data, generator_cls):
        gen = generator_cls(random_state=42)
        gen.fit(
            sample_data,
            categorical_columns=["department"],
            continuous_columns=["age", "salary"],
        )
        assert gen._is_fitted

        synth = gen.generate(n_samples=50)
        assert isinstance(synth, pd.DataFrame)
        assert len(synth) == 50
        # All original columns present
        for col in sample_data.columns:
            assert col in synth.columns

    def test_generate_before_fit_raises(self, generator_cls):
        gen = generator_cls()
        with pytest.raises(RuntimeError, match="must be fitted"):
            gen.generate(n_samples=10)

    def test_deterministic_with_same_seed(self, sample_data, generator_cls):
        gen1 = generator_cls(random_state=123)
        gen1.fit(
            sample_data, categorical_columns=["department"], continuous_columns=["age", "salary"]
        )
        out1 = gen1.generate(50)

        gen2 = generator_cls(random_state=123)
        gen2.fit(
            sample_data, categorical_columns=["department"], continuous_columns=["age", "salary"]
        )
        out2 = gen2.generate(50)

        pd.testing.assert_frame_equal(out1, out2)

    def test_repr(self, generator_cls):
        gen = generator_cls()
        assert "GaussianCopulaGenerator" in repr(gen)
