"""Tests for metis.infrastructure.preprocess.caster module."""

import pandas as pd
import pytest

from metis.infrastructure.preprocess.caster import SimpleCaster


@pytest.fixture
def basic_schema():
    """A simple schema for testing."""
    return {
        "columns": {
            "age": "continuous",
            "gender": "categorical",
            "city": "categorical",
            "score": "continuous",
        }
    }


@pytest.fixture
def schema_with_ordinal():
    """Schema with ordinal column."""
    return {
        "columns": {
            "age": "continuous",
            "education": {"type": "ordinal", "levels": ["low", "medium", "high"]},
            "gender": "categorical",
        }
    }


@pytest.fixture
def schema_with_id():
    """Schema that excludes an ID column."""
    return {
        "columns": {
            "patient_id": "id",
            "age": "continuous",
            "gender": "categorical",
        }
    }


@pytest.fixture
def sample_df():
    """Small sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 45.0, 55.0, 60.0],
            "gender": ["M", "F", "M", "F", "M"],
            "city": ["Madrid", "Barcelona", "Valencia", "Madrid", "Sevilla"],
            "score": [0.8, 0.6, 0.9, 0.4, 0.7],
        }
    )


class TestSimpleCasterBasic:
    """Tests for basic SimpleCaster functionality."""

    def test_normalize_str(self):
        assert SimpleCaster._normalize_str("  Hello ") == "hello"
        assert SimpleCaster._normalize_str("WORLD") == "world"
        assert SimpleCaster._normalize_str(None) is None

    def test_is_num(self):
        assert SimpleCaster._is_num(pd.Series([1, 2, 3])) is True
        assert SimpleCaster._is_num(pd.Series([1.0, 2.0])) is True
        assert SimpleCaster._is_num(pd.Series(["a", "b"])) is False

    def test_fit_transform_continuous(self, sample_df, basic_schema):
        from metis.domain.contracts import TypeSchema

        schema = TypeSchema(columns=basic_schema["columns"])
        caster = SimpleCaster(schema)
        caster.fit(sample_df)
        cat, num = caster.transform(sample_df)

        # Should have numeric data for continuous columns
        assert "age" in num.columns
        assert "score" in num.columns

    def test_fit_transform_categorical(self, sample_df, basic_schema):
        from metis.domain.contracts import TypeSchema

        schema = TypeSchema(columns=basic_schema["columns"])
        caster = SimpleCaster(schema)
        caster.fit(sample_df)
        cat, num = caster.transform(sample_df)

        # Should have categorical data
        assert "gender" in cat.columns
        assert "city" in cat.columns

    def test_id_columns_excluded(self):
        from metis.domain.contracts import TypeSchema

        df = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "age": [25.0, 30.0, 45.0],
                "gender": ["M", "F", "M"],
            }
        )
        schema = TypeSchema(
            columns={"patient_id": "id", "age": "continuous", "gender": "categorical"}
        )
        caster = SimpleCaster(schema)
        caster.fit(df)
        cat, num = caster.transform(df)

        # id columns may still appear in num (cast to numeric) but should
        # not appear in cat output
        assert "patient_id" not in cat.columns

    def test_ordinal_to_numeric(self):
        from metis.domain.contracts import TypeSchema

        df = pd.DataFrame(
            {
                "education": ["low", "medium", "high", "low", "high"],
            }
        )
        schema = TypeSchema(
            columns={"education": {"type": "ordinal", "levels": ["low", "medium", "high"]}}
        )
        caster = SimpleCaster(schema)
        caster.fit(df)
        cat, num = caster.transform(df)

        # Ordinal columns get __ord suffix
        ord_cols = [c for c in num.columns if "education" in c]
        assert len(ord_cols) == 1
        values = num[ord_cols[0]].values
        assert values.min() >= 0.0
        assert values.max() <= 1.0
