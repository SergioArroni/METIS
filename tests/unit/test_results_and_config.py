"""Tests for metis.shared.results and metis.shared.config_utils.

WHY results: ColumnMetricResult is the data carrier between metric computation
and aggregation. Broken serialization = lost evaluation data.

WHY config_utils: none_safe() silently converts "None" strings from YAML
into Python None. Without it, target="None" would try to find a column
literally named "None".
"""

from metis.shared.config_utils import none_safe
from metis.shared.results import ColumnMetricResult

# =============================================================================
# ColumnMetricResult
# =============================================================================


class TestColumnMetricResult:
    def test_creation(self):
        r = ColumnMetricResult(column="age", raw_value=0.05, normalized_value=0.95, is_valid=True)
        assert r.column == "age"
        assert r.raw_value == 0.05
        assert r.normalized_value == 0.95
        assert r.is_valid is True
        assert r.error is None

    def test_to_dict(self):
        r = ColumnMetricResult(column="income", raw_value=0.3, normalized_value=0.7, is_valid=True)
        d = r.to_dict()
        assert d["column"] == "income"
        assert d["raw_value"] == 0.3
        assert d["normalized_value"] == 0.7
        assert d["is_valid"] is True
        assert d["error"] is None

    def test_from_dict_roundtrip(self):
        original = ColumnMetricResult(
            column="x", raw_value=0.1, normalized_value=0.9, is_valid=True, error=None
        )
        restored = ColumnMetricResult.from_dict(original.to_dict())
        assert restored.column == original.column
        assert restored.raw_value == original.raw_value
        assert restored.normalized_value == original.normalized_value
        assert restored.is_valid == original.is_valid

    def test_invalid_factory(self):
        r = ColumnMetricResult.invalid("broken_col", "Division by zero")
        assert r.column == "broken_col"
        assert r.is_valid is False
        assert r.error == "Division by zero"
        assert r.normalized_value == 0.0

    def test_from_dict_with_error(self):
        data = {
            "column": "c",
            "raw_value": 0.0,
            "normalized_value": 0.0,
            "is_valid": False,
            "error": "timeout",
        }
        r = ColumnMetricResult.from_dict(data)
        assert r.error == "timeout"
        assert r.is_valid is False


# =============================================================================
# none_safe
# =============================================================================


class TestNoneSafe:
    """Risk: YAML loads 'None' as string. Without none_safe, the pipeline
    would look for a column literally named 'None'."""

    def test_none_string_converted(self):
        assert none_safe("None") is None

    def test_case_insensitive(self):
        assert none_safe("none") is None
        assert none_safe("NONE") is None
        assert none_safe("  None  ") is None

    def test_actual_none_passthrough(self):
        assert none_safe(None) is None

    def test_regular_string_passthrough(self):
        assert none_safe("income") == "income"

    def test_numeric_passthrough(self):
        assert none_safe(42) == 42
        assert none_safe(3.14) == 3.14

    def test_empty_string_passthrough(self):
        assert none_safe("") == ""

    def test_list_passthrough(self):
        val = ["a", "b"]
        assert none_safe(val) is val
