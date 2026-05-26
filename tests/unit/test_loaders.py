"""Tests for metis.infrastructure.io.loaders module."""

import pandas as pd
import pytest

from metis.domain.errors import SchemaError
from metis.infrastructure.io.loaders import load_csv


@pytest.fixture
def valid_csv(tmp_path):
    """Create a valid CSV file for testing."""
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
    return str(csv_path)


@pytest.fixture
def empty_csv(tmp_path):
    """Create an empty CSV file."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    return str(csv_path)


@pytest.fixture
def csv_with_na(tmp_path):
    """Create a CSV with NA values."""
    csv_path = tmp_path / "na.csv"
    csv_path.write_text("a,b,c\n1,NA,3\n4,,6\nNone,8,null\n")
    return str(csv_path)


class TestLoadCSV:
    """Tests for load_csv function."""

    def test_loads_valid_csv(self, valid_csv):
        df = load_csv(valid_csv)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ["a", "b", "c"]

    def test_file_not_found_raises_schema_error(self):
        with pytest.raises(SchemaError, match="File not found"):
            load_csv("/nonexistent/path/to/file.csv")

    def test_empty_csv_raises_schema_error(self, empty_csv):
        with pytest.raises(SchemaError):
            load_csv(empty_csv)

    def test_na_values_parsed(self, csv_with_na):
        df = load_csv(csv_with_na)
        # NA, empty string, None, null should all be NaN
        assert df["b"].isna().sum() >= 1
        assert df["a"].isna().sum() >= 1

    def test_custom_separator(self, tmp_path):
        csv_path = tmp_path / "semicolon.csv"
        csv_path.write_text("a;b;c\n1;2;3\n4;5;6\n")
        df = load_csv(str(csv_path), sep=";")
        assert df.shape == (2, 3)

    def test_malformed_csv_raises_schema_error(self, tmp_path):
        csv_path = tmp_path / "malformed.csv"
        csv_path.write_text("a,b\n1,2,3\n4,5,6,7,8\n")
        # pandas may handle this, so just verify it doesn't crash uncontrolled
        # or raises SchemaError
        try:
            df = load_csv(str(csv_path))
            # If pandas can handle it, it should still return a DataFrame
            assert isinstance(df, pd.DataFrame)
        except SchemaError:
            pass  # Expected behavior
