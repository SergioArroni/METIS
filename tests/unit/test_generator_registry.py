"""Unit tests for generator registry including new generators."""

import pytest

from metis.sota_models.generators.registry import GeneratorRegistry


class TestGeneratorRegistry:
    """Tests for GeneratorRegistry registration and creation."""

    def test_gaussian_copula_is_registered(self):
        assert "gaussian_copula" in GeneratorRegistry.available()

    def test_create_gaussian_copula(self):
        gen = GeneratorRegistry.create("gaussian_copula", random_state=42)
        assert gen.name == "Gaussian Copula"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown generator"):
            GeneratorRegistry.create("nonexistent_model")

    def test_all_default_generators_registered(self):
        available = GeneratorRegistry.available()
        expected = {
            "smotenc",
            "bootstrap",
            "delete_zero",
            "delete_mean",
            "uniform_noise",
            "real_data",
            "gaussian_copula",
            "bn",
            "bayesian_network",
            "cart",
            "ctgan",
        }
        assert expected.issubset(set(available))

    def test_adsgan_is_not_a_silent_ctgan_alias(self):
        assert "adsgan" not in GeneratorRegistry.available()
        from metis.sota_models.generators.adsgan import ADSGANGenerator

        with pytest.raises(NotImplementedError, match="ADS-GAN"):
            ADSGANGenerator()
