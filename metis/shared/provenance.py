"""Which METIS produced a result: version and, from a git checkout, the commit.

``metis-eval`` kept the version 0.1.0 across every change, so a run could
not say which code scored it.  ``RunSummary.artifacts`` now carries both.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def metis_provenance() -> dict[str, str | None]:
    """``{"metis_version": ..., "metis_commit": ...}``; the commit is None outside git.

    A checkout with uncommitted changes gets a ``-dirty`` suffix.
    """
    from metis import __version__  # noqa: PLC0415  (metis/__init__ imports nothing heavy)

    root = Path(__file__).resolve().parents[2]
    commit: str | None = None
    try:
        commit = (
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            or None
        )
        if commit:
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            if dirty:
                commit += "-dirty"
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"metis_version": __version__, "metis_commit": commit}
