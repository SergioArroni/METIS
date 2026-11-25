"""Every third-party module the evaluation imports is a base dependency.

``catboost`` was only in the ``ml`` extra although the metric registry
imports it: a plain install (and CI, which installs ``.[ci]``) could not
``import metis``.  ``metis.sota_models`` (benchmark generators) is excluded;
optional imports guarded by ``try/except ImportError`` are listed below.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL = {"torch", "tensorflow"}  # guarded in metis/shared/reproducibility.py
DIST_NAME = {"sklearn": "scikit-learn", "yaml": "pyyaml"}


def _imported_modules() -> set[str]:
    mods: set[str] = set()
    for path in (ROOT / "metis").rglob("*.py"):
        if "sota_models" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    return {m for m in mods if m not in sys.stdlib_module_names and m != "metis"}


def test_core_imports_are_base_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {
        re.split(r"[<>=!~\[ ;]", dep, maxsplit=1)[0].lower() for dep in project["dependencies"]
    }
    missing = sorted(
        DIST_NAME.get(m, m)
        for m in _imported_modules() - OPTIONAL
        if DIST_NAME.get(m, m) not in declared
    )
    assert not missing, f"imported by metis but not in [project].dependencies: {missing}"


def test_sdk_import_errors_are_not_swallowed(monkeypatch):
    import builtins
    import importlib

    import pytest

    import metis

    real_import = builtins.__import__

    def _no_sdk(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("interface") and fromlist and "sdk" in fromlist:
            raise ImportError("No module named 'catboost'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_sdk)
    with pytest.raises(ImportError, match="catboost"):
        metis.Evaluator  # noqa: B018
    monkeypatch.undo()
    assert importlib.import_module("metis").Evaluator.__name__ == "Evaluator"
