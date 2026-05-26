"""Automatic semantic version bumping for CI.

Usage:
    python scripts/bump_version.py auto --branch <branch_name>

Rules:
    - push to main     → bump minor  (1.0.0 → 1.1.0)
    - push to staging  → bump patch  (1.0.0 → 1.0.1)
    - push to development → bump dev suffix (1.0.0 → 1.0.1.dev1)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parent.parent / "version.py"
VERSION_RE = re.compile(
    r'^__version__\s*=\s*["\']'
    r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:\.dev(?P<dev>\d+))?"
    r'["\']',
    re.MULTILINE,
)


def read_version() -> tuple[int, int, int, int | None]:
    content = VERSION_FILE.read_text(encoding="utf-8")
    m = VERSION_RE.search(content)
    if not m:
        print(f"ERROR: Cannot parse version in {VERSION_FILE}", file=sys.stderr)
        sys.exit(1)
    major = int(m.group("major"))
    minor = int(m.group("minor"))
    patch = int(m.group("patch"))
    dev = int(m.group("dev")) if m.group("dev") else None
    return major, minor, patch, dev


def write_version(major: int, minor: int, patch: int, dev: int | None = None) -> str:
    if dev is not None:
        version_str = f"{major}.{minor}.{patch}.dev{dev}"
    else:
        version_str = f"{major}.{minor}.{patch}"
    VERSION_FILE.write_text(f'__version__ = "{version_str}"\n', encoding="utf-8")
    return version_str


def bump(branch: str) -> str:
    major, minor, patch, dev = read_version()

    if branch == "main":
        # Minor bump, reset patch
        return write_version(major, minor + 1, 0)
    if branch == "staging":
        # Patch bump
        return write_version(major, minor, patch + 1)
    if branch == "development":
        # Dev suffix increment
        new_dev = (dev or 0) + 1
        return write_version(major, minor, patch, new_dev)
    print(f"Unknown branch '{branch}', no bump applied.", file=sys.stderr)
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump version automatically")
    parser.add_argument("action", choices=["auto"], help="Bump action")
    parser.add_argument("--branch", required=True, help="Git branch name")
    args = parser.parse_args()

    new_version = bump(args.branch)
    print(f"Bumped to {new_version}")


if __name__ == "__main__":
    main()
