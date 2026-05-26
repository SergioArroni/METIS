"""Tests for metis.shared.schema_utils module."""

from metis.shared.schema_utils import extract_column_types, filter_schema_columns


class TestExtractColumnTypes:
    """Tests for extract_column_types."""

    def test_basic_string_types(self):
        schema = {
            "age": "continuous",
            "gender": "categorical",
            "patient_id": "id",
        }
        result = extract_column_types(schema)
        assert "age" in result["continuous"]
        assert "gender" in result["categorical"]
        assert "patient_id" in result["id"]

    def test_dict_type_spec(self):
        schema = {
            "income": {"type": "discrete", "ranges": [[0, 1000], [1001, 5000]]},
            "education": {"type": "ordinal", "levels": ["low", "medium", "high"]},
        }
        result = extract_column_types(schema)
        assert "income" in result["continuous"]  # discrete → continuous bucket
        assert "education" in result["ordinal"]
        assert result["ordinal"]["education"] == ["low", "medium", "high"]

    def test_boolean_goes_to_categorical(self):
        schema = {"has_insurance": "boolean"}
        result = extract_column_types(schema)
        assert "has_insurance" in result["categorical"]

    def test_text_goes_to_categorical(self):
        schema = {"description": "text"}
        result = extract_column_types(schema)
        assert "description" in result["categorical"]

    def test_code_numeric_goes_to_categorical(self):
        schema = {"zip_code": "code_numeric"}
        result = extract_column_types(schema)
        assert "zip_code" in result["categorical"]

    def test_datetime_goes_to_continuous(self):
        schema = {"created_at": "datetime"}
        result = extract_column_types(schema)
        assert "created_at" in result["continuous"]

    def test_geospatial_goes_to_continuous(self):
        schema = {"lat": "geospatial"}
        result = extract_column_types(schema)
        assert "lat" in result["continuous"]

    def test_unknown_type_defaults_to_continuous(self):
        schema = {"mystery": "unknown_type"}
        result = extract_column_types(schema)
        assert "mystery" in result["continuous"]

    def test_empty_schema(self):
        result = extract_column_types({})
        assert result == {"categorical": [], "ordinal": {}, "continuous": [], "id": []}

    def test_mixed_schema(self):
        schema = {
            "id_col": "id",
            "age": "continuous",
            "name": "categorical",
            "level": {"type": "ordinal", "levels": ["a", "b", "c"]},
            "flag": "boolean",
        }
        result = extract_column_types(schema)
        assert len(result["id"]) == 1
        assert len(result["continuous"]) == 1
        assert len(result["categorical"]) == 2  # name + flag (boolean)
        assert len(result["ordinal"]) == 1


class TestFilterSchemaColumns:
    """Tests for filter_schema_columns."""

    def test_removes_id_columns(self):
        schema = {
            "patient_id": "id",
            "age": "continuous",
            "gender": "categorical",
        }
        filtered = filter_schema_columns(schema)
        assert "patient_id" not in filtered
        assert "age" in filtered
        assert "gender" in filtered

    def test_keeps_all_non_id(self):
        schema = {
            "a": "continuous",
            "b": "categorical",
            "c": {"type": "ordinal", "levels": ["x", "y"]},
        }
        filtered = filter_schema_columns(schema)
        assert len(filtered) == 3

    def test_dict_type_id_removed(self):
        schema = {
            "row_id": {"type": "id"},
            "value": "continuous",
        }
        filtered = filter_schema_columns(schema)
        assert "row_id" not in filtered
        assert "value" in filtered

    def test_empty_schema(self):
        assert filter_schema_columns({}) == {}
