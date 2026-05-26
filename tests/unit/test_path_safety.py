"""Tests for metis.shared.path_safety — path traversal protection.

WHY: **SECURITY-CRITICAL**. This module prevents path traversal attacks
when user-supplied config values (output_dir, bounds_file, etc.) are used
to construct filesystem paths. A bypass here = arbitrary file read/write.
"""

import pytest

from metis.shared.path_safety import UnsafePathError, ensure_within, safe_join

# =============================================================================
# safe_join — main defense against path traversal
# =============================================================================


class TestSafeJoin:
    """Test safe_join rejects all traversal vectors."""

    def test_simple_relative_path(self, tmp_path):
        result = safe_join(tmp_path, "output", "report.json")
        assert str(result).startswith(str(tmp_path.resolve()))
        assert result.name == "report.json"

    def test_nested_relative_path(self, tmp_path):
        result = safe_join(tmp_path, "a", "b", "c.txt")
        assert result == tmp_path.resolve() / "a" / "b" / "c.txt"

    # --- Traversal attacks ---

    def test_rejects_dotdot(self, tmp_path):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, "..", "etc", "passwd")

    def test_rejects_embedded_dotdot(self, tmp_path):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, "a", "b", "..", "..", "..", "secret")

    def test_rejects_absolute_unix_path(self, tmp_path):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, "/etc/passwd")

    def test_rejects_absolute_windows_path(self, tmp_path):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, "C:\\Windows\\System32")

    def test_rejects_nul_byte(self, tmp_path):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, "file\x00.txt")

    def test_rejects_backslash_prefix(self, tmp_path):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, "\\server\\share")

    def test_rejects_forward_slash_prefix(self, tmp_path):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, "/absolute/path")

    # --- Edge cases ---

    def test_empty_parts(self, tmp_path):
        """Empty string parts should resolve to root itself."""
        result = safe_join(tmp_path, "")
        assert result == tmp_path.resolve()

    def test_single_filename(self, tmp_path):
        result = safe_join(tmp_path, "report.json")
        assert result.name == "report.json"

    def test_path_with_spaces(self, tmp_path):
        result = safe_join(tmp_path, "my folder", "my file.txt")
        assert "my folder" in str(result)


# =============================================================================
# ensure_within — validates pre-existing paths
# =============================================================================


class TestEnsureWithin:
    def test_valid_containment(self, tmp_path):
        child = tmp_path / "sub" / "file.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        result = ensure_within(tmp_path, child)
        assert result == child.resolve()

    def test_escape_rejected(self, tmp_path):
        outside = tmp_path.parent / "sibling"
        with pytest.raises(UnsafePathError):
            ensure_within(tmp_path, outside)

    def test_same_directory(self, tmp_path):
        """Root itself should be valid."""
        result = ensure_within(tmp_path, tmp_path)
        assert result == tmp_path.resolve()


# =============================================================================
# UnsafePathError
# =============================================================================


class TestUnsafePathError:
    def test_is_value_error(self):
        assert issubclass(UnsafePathError, ValueError)

    def test_message_informative(self):
        err = UnsafePathError("path /foo escapes /bar")
        assert "/foo" in str(err)
