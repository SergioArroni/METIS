"""The transversal axis: does the synthetic table respect the rules of its domain?

Two sources of rules, both declared, never estimated from the data:

- the schema: what the user already said when typing the columns becomes a
  rule with no extra work (the levels of an ordinal, the levels or the step
  of a discrete, declared categories of a nominal);
- ``data.constraints.rules``: domain knowledge and physical laws the user
  adds (``ap_lo <= ap_hi``, ``IF sex == male THEN pregnancies == 0``,
  ``bmi = weight / height**2``).

Each rule is checked on every synthetic row.  The result is a proportion,
not a score: the share of the rows the rule applies to that break it, with
its 95 % Wilson interval, and a verdict (``ok`` while that share is at most
``tolerance``, 0 by default).  The same rules are checked on the real table:
a rule the real data break is wrongly declared or the data are dirty, and
the report says so instead of blaming the generator.  None of this enters
the fidelity, utility or privacy scores; the evaluation goes on.

Configuration (``data.constraints``)::

    constraints:
      missing: not_applicable     # or violates / any_category
      tolerance: 0.0              # largest violation share still "ok"
      from_schema: true           # rules from the schema (default true)
      rules:
        - compare: {left: ap_lo, op: "<=", right: ap_hi}
        - range: {column: age, low: 0, high: 120}
        - conditional:
            if_column: sex
            if_values: [male]
            then: [{allowed: {column: pregnancies, values: [0]}}]
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ...domain.errors import ConfigError
from .rules import MISSING_SEMANTICS, Allowed, Constraints, Grid, Rule, rules_from_list

_EXAMPLES = 5
_KNOWN = {"missing", "tolerance", "from_schema", "rules", "domains"}


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval of a proportion k/n (valid at 0 and n)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def rules_from_schema(schema: dict[str, Any]) -> list[Rule]:
    """The rules the typing already declares (nothing is read from the data)."""
    out: list[Rule] = []
    for col, spec in (schema or {}).items():
        if not isinstance(spec, dict):
            continue
        typ, levels, step = spec.get("type"), spec.get("levels"), spec.get("step")
        if typ in ("ordinal", "nominal") and levels:
            out.append(Allowed(col, tuple(levels), name=f"{col}: declared {typ} levels"))
        elif typ == "discrete" and levels:
            out.append(Allowed(col, tuple(levels), name=f"{col}: declared discrete values"))
        elif typ == "discrete" and step:  # a lattice not anchored at 0 goes in data.constraints
            out.append(Grid(col, float(step), name=f"{col}: step {step}"))
    return out


def build_constraints(config: dict[str, Any]) -> tuple[Constraints, dict[str, str], float] | None:
    """``(constraints, origin of each rule label, tolerance)``, or None when
    there is nothing to check.  Malformed declarations raise ConfigError."""
    data = config.get("data") or {}
    raw = data.get("constraints") or {}
    if not isinstance(raw, dict):
        raise ConfigError("'data.constraints' must be a mapping (missing, tolerance, from_schema, rules)")
    unknown = set(raw) - _KNOWN
    if unknown:
        raise ConfigError(f"'data.constraints': unknown key(s) {sorted(unknown)}; known: {sorted(_KNOWN)}")
    missing = raw.get("missing", "not_applicable")
    if missing not in MISSING_SEMANTICS:
        raise ConfigError(f"'data.constraints.missing' must be one of {MISSING_SEMANTICS}, got {missing!r}")
    tolerance = float(raw.get("tolerance", 0.0))
    if not 0.0 <= tolerance < 1.0:
        raise ConfigError(f"'data.constraints.tolerance' is a share in [0, 1), got {tolerance}")
    schema = data.get("schema") if isinstance(data.get("schema"), dict) else {}
    try:
        user = list(rules_from_list(raw.get("rules") or []))
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"'data.constraints.rules': {e}") from e
    auto = rules_from_schema(schema) if raw.get("from_schema", True) else []
    rules = auto + user
    if not rules:
        return None
    origin = {}
    for r in auto:
        origin[r.label()] = "schema"
    for r in user:
        origin.setdefault(r.label(), "user")
    domains = {k: tuple(v) for k, v in (raw.get("domains") or {}).items()}
    return Constraints(tuple(rules), missing, domains), origin, tolerance


def _side(constraints: Constraints, rule: Rule, table: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray]:
    o = constraints.outcome(rule, table)
    n, k = int(o.checked.sum()), int(o.violated.sum())
    lo, hi = wilson_interval(k, n)
    return {
        "rows": int(len(table)),
        "rows_checked": n,
        "violations": k,
        "violation_rate": (k / n) if n else None,
        "violation_rate_ci95": [lo, hi] if n else None,
    }, o.violated


def _plain(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA or v is pd.NaT:
        return None
    return v.item() if isinstance(v, np.generic) else v


def _examples(table: pd.DataFrame, cols: list[str], violated: np.ndarray) -> list[dict[str, Any]]:
    """The first violating rows, with their position and the values the rule reads."""
    return [
        {"row": int(i), **{c: _plain(table[c].iat[i]) for c in cols}}
        for i in np.flatnonzero(violated)[:_EXAMPLES]
    ]


def validate(
    real: pd.DataFrame, synth: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    """The constraints report and its warnings (None when nothing is declared)."""
    built = build_constraints(config)
    if built is None:
        return None, []
    constraints, origin, tolerance = built
    for rule in constraints.rules:
        absent = [c for c in rule.columns() if c not in synth.columns]
        if absent:
            raise ConfigError(f"constraint {rule.label()!r}: column(s) {absent} not in the synthetic table")
    rules_out, warnings = [], []
    any_synth = np.zeros(len(synth), bool)
    for rule in constraints.rules:
        label = rule.label()
        s, viol = _side(constraints, rule, synth)
        any_synth |= viol
        r = _side(constraints, rule, real)[0] if all(c in real.columns for c in rule.columns()) else None
        rate = s["violation_rate"]
        ok = rate is None or rate <= tolerance
        cols = list(dict.fromkeys(rule.columns()))
        entry = {
            "rule": label,
            "kind": rule.kind,
            "origin": origin.get(label, "user"),
            "columns": cols,
            "ok": bool(ok),
            "compliance": (1.0 - rate) if rate is not None else None,
            "synthetic": s,
            "real": r,
            "examples": _examples(synth, cols, viol),
        }
        rules_out.append(entry)
        if not ok:
            warnings.append(
                f"constraint violated: {label} — {s['violations']} of {s['rows_checked']} synthetic "
                f"rows ({rate:.1%})"
            )
        if r is not None and r["violations"]:
            warnings.append(
                f"the real table also breaks {label!r} ({r['violations']} of {r['rows_checked']} rows): "
                "the rule may be wrongly declared or the real data dirty"
            )
    n_failed = sum(not e["ok"] for e in rules_out)
    report = {
        "ok": n_failed == 0,
        "n_rules": len(rules_out),
        "n_failed": n_failed,
        "missing": constraints.missing,
        "tolerance": tolerance,
        "synthetic_rows_violating_any": int(any_synth.sum()),
        "synthetic_rows_violating_any_rate": float(any_synth.mean()) if len(synth) else None,
        "rules": rules_out,
    }
    return report, warnings
