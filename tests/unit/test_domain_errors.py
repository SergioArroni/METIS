"""Tests for metis.domain.errors — exception hierarchy.

WHY: Incorrect exception attributes break error handlers in CLI/SDK layers.
Each error must carry its specific context (config_path, column, original_error).
"""

import pytest

from metis.domain.errors import (
    ConfigError,
    METISError,
    PreprocessingError,
    RegistryError,
    SchemaError,
    TypeCastingError,
)


class TestErrorHierarchy:
    """All domain errors inherit from METISError for unified catching."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigError,
            SchemaError,
            RegistryError,
            PreprocessingError,
            TypeCastingError,
        ],
    )
    def test_inherits_from_metis_error(self, exc_class):
        assert issubclass(exc_class, METISError)
        assert issubclass(exc_class, Exception)


class TestConfigError:
    def test_message_formatting(self):
        err = ConfigError("missing key")
        assert "Configuration error: missing key" in str(err)
        assert err.config_path is None

    def test_with_config_path(self):
        err = ConfigError("bad value", config_path="config.yaml")
        assert "config.yaml" in str(err)
        assert err.config_path == "config.yaml"

    def test_catchable_as_metis_error(self):
        with pytest.raises(METISError):
            raise ConfigError("test")


class TestSchemaError:
    def test_message_formatting(self):
        err = SchemaError("type mismatch")
        assert "Schema error" in str(err)
        assert err.column is None

    def test_with_column(self):
        err = SchemaError("invalid type", column="age")
        assert "age" in str(err)
        assert err.column == "age"


class TestRegistryError:
    def test_minimal(self):
        err = RegistryError("not found")
        assert "Registry error" in str(err)
        assert err.registry_type is None
        assert err.item_id is None

    def test_full_context(self):
        err = RegistryError("not found", registry_type="metric", item_id="fidelity.ks")
        assert "metric registry" in str(err)
        assert "fidelity.ks" in str(err)
        assert err.registry_type == "metric"
        assert err.item_id == "fidelity.ks"


class TestPreprocessingError:
    def test_minimal(self):
        err = PreprocessingError("failed")
        assert "Preprocessing error" in str(err)
        assert err.step is None
        assert err.original_error is None

    def test_with_step_and_cause(self):
        cause = ValueError("bad column")
        err = PreprocessingError("cast failed", step="type_casting", original_error=cause)
        assert "type_casting" in str(err)
        assert "ValueError" in str(err)
        assert err.step == "type_casting"
        assert err.original_error is cause


class TestTypeCastingError:
    def test_full_context(self):
        cause = TypeError("cannot convert")
        err = TypeCastingError(
            "conversion failed",
            column="salary",
            expected_type="float",
            original_error=cause,
        )
        assert "salary" in str(err)
        assert "float" in str(err)
        assert "TypeError" in str(err)
        assert err.column == "salary"
        assert err.expected_type == "float"
        assert err.original_error is cause
