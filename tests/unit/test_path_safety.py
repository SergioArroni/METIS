"""Tests for ``metis.shared.path_safety.safe_join`` / ``ensure_within``."""

from __future__ import annotations

import pytest

from metis.shared.path_safety import UnsafePathError, ensure_within, safe_join


def test_safe_join_basic(tmp_path):
    assert safe_join(tmp_path, "sub", "file.txt") == (tmp_path / "sub" / "file.txt").resolve()


def test_safe_join_rejects_parent_traversal(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "..", "etc", "passwd")


def test_safe_join_rejects_absolute_part(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "/etc/passwd")


def test_safe_join_rejects_windows_absolute(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "C:\\Windows\\System32")


def test_safe_join_rejects_nul_byte(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "evil\x00.csv")


def test_safe_join_keeps_inner_traversal(tmp_path):
    # a/b/.. resolves back to a — this is OK, still inside root.
    result = safe_join(tmp_path, "a", "b", "..", "c.txt")
    assert result == (tmp_path / "a" / "c.txt").resolve()


def test_ensure_within_accepts_inside(tmp_path):
    inside = tmp_path / "x" / "y.txt"
    assert ensure_within(tmp_path, inside) == inside.resolve()


def test_ensure_within_rejects_outside(tmp_path):
    with pytest.raises(UnsafePathError):
        ensure_within(tmp_path, tmp_path.parent / "elsewhere.txt")
