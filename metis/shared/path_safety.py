"""Path-handling helpers with guards against traversal and unsafe joins.

These helpers are intended for any code that builds output/input paths from
user-provided configuration values (YAML keys such as ``output_dir``,
``data.real``, ``calibration.bounds_file``...).

Design goals
------------
* Reject inputs containing ``..`` segments, NUL bytes, or absolute paths
  that would escape the configured root.
* Resolve symlinks and verify the result is still inside the root after
  normalisation (defends against ``a/b/../../etc/passwd`` and similar).
* Be operating-system aware: on Windows ``C:\\`` and forward/back-slashes
  are both treated as absolute.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePath, PureWindowsPath


class UnsafePathError(ValueError):
    """Raised when a candidate path would escape the configured root."""


def safe_join(root: str | os.PathLike[str], *parts: str | os.PathLike[str]) -> Path:
    """Join *parts* under *root*, refusing any result that escapes *root*.

    Parameters
    ----------
    root:
        Trusted directory. Resolved to an absolute path; created on demand
        is the caller's responsibility.
    parts:
        Untrusted segments to append.

    Returns
    -------
    Path
        Absolute, normalised path guaranteed to live inside ``root``.

    Raises
    ------
    UnsafePathError
        If any part is absolute, contains NUL bytes, or the resolved path
        falls outside ``root``.
    """
    root_path = Path(root).resolve()

    for part in parts:
        as_str = os.fspath(part)
        if "\x00" in as_str:
            raise UnsafePathError(f"Path part contains NUL byte: {as_str!r}")
        if (
            PurePath(as_str).is_absolute()
            or PureWindowsPath(as_str).is_absolute()
            or as_str.startswith(("/", "\\"))
        ):
            raise UnsafePathError(f"Path part must be relative, got absolute: {as_str!r}")

    candidate = root_path.joinpath(*[os.fspath(p) for p in parts])
    # Resolve without requiring the path to exist (strict=False) so we can
    # validate output paths before they are created.
    resolved = candidate.resolve()

    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise UnsafePathError(f"Resolved path {resolved!s} escapes root {root_path!s}") from exc

    return resolved


def ensure_within(root: str | os.PathLike[str], candidate: str | os.PathLike[str]) -> Path:
    """Validate that an already-built ``candidate`` lives under ``root``.

    Useful when the path is constructed elsewhere (e.g. from existing files
    discovered via ``glob``) and only needs a containment check.
    """
    root_path = Path(root).resolve()
    cand_path = Path(candidate).resolve()
    try:
        cand_path.relative_to(root_path)
    except ValueError as exc:
        raise UnsafePathError(f"Path {cand_path!s} is not inside root {root_path!s}") from exc
    return cand_path
