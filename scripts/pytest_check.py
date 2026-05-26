"""Pre-commit hook: run pytest and always exit 0 (non-blocking)."""

import pathlib
import subprocess
import sys


def _find_venv_python() -> str:
    """Locate the venv python executable."""
    root = pathlib.Path(__file__).resolve().parent.parent
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def main() -> int:
    python = _find_venv_python()
    result = subprocess.run(
        [python, "-m", "pytest", "tests/", "-q", "--tb=line", "--no-header"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="")
        print("[pytest-check] Tests failed but commit is not blocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
