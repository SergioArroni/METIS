"""Rules a table must satisfy, checked row by row.

A rule is domain knowledge or a physical law that holds for every record:
a value inside its range or on its lattice, a level that exists, ``ap_lo <=
ap_hi``, ``IF sex == male THEN pregnancies == 0``.  Checking one never
estimates anything: each row satisfies it or not.

Every rule returns a :class:`Check` per row: whether the rule *applies* to
the row, whether it is *violated*, and whether it could not be checked
because a value it needs is *missing*.  What a missing value means is
declared once for the whole set (:attr:`Constraints.missing`), because the
reported violation rate depends on it:

- ``"not_applicable"``: a rule that needs a missing value does not fire;
- ``"violates"``: a rule that needs a missing value is violated;
- ``"any_category"``: a missing *condition* of a :class:`Conditional` is
  valid if the row satisfies the consequent of some value of that column;
  any other missing value is treated as ``"not_applicable"``.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

MISSING_SEMANTICS = ("not_applicable", "violates", "any_category")
_TOL = 1e-9


# ---------------------------------------------------------------------- values
def _canonical(v: Any) -> str:
    """The same value written as ``1``, ``1.0`` or ``" 1 "`` is one value."""
    from metis.infrastructure.preprocess.caster import SimpleCaster

    return SimpleCaster._normalize_str(v)


def is_missing(values: pd.Series) -> np.ndarray:
    return values.isna().to_numpy()


def _numeric(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(float values, NaN where missing or unreadable; missing mask)."""
    miss = is_missing(values)
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return x, miss


def _isin(values: pd.Series, allowed: Iterable[Any]) -> np.ndarray:
    """Element-wise membership: numerically when both sides are numbers, else
    on the canonical strings (``1``, ``1.0`` and ``"1"`` match)."""
    allowed = list(allowed)
    nums = [a for a in allowed if isinstance(a, (int, float, np.number)) and not isinstance(a, bool)]
    if pd.api.types.is_numeric_dtype(values) and allowed and len(nums) == len(allowed):
        x = values.to_numpy(dtype=float)
        ref = np.asarray(nums, dtype=float)
        return np.any(np.isclose(x[:, None], ref[None, :], rtol=0.0, atol=_TOL), axis=1)
    keys = {_canonical(a) for a in allowed}
    return values.map(lambda v: not pd.isna(v) and _canonical(v) in keys).to_numpy(dtype=bool)


def _column(table: pd.DataFrame, name: str) -> pd.Series:
    if name not in table.columns:
        raise KeyError(f"column {name!r} is not in the table")
    return table[name]


# ----------------------------------------------------------------------- rules
@dataclass(frozen=True)
class Check:
    """A rule on a table, per row.  ``missing_condition`` marks the rows whose
    :class:`Conditional` could not fire because its condition is missing."""

    applies: np.ndarray
    violated: np.ndarray
    missing: np.ndarray
    missing_condition: np.ndarray | None = None


def _everywhere(violated: np.ndarray, missing: np.ndarray) -> Check:
    return Check(np.ones(len(violated), bool), violated & ~missing, missing)


@dataclass(frozen=True)
class Range:
    """``low <= column <= high`` (either bound may be None)."""

    column: str
    low: float | None = None
    high: float | None = None
    name: str | None = None
    kind = "range"

    def columns(self) -> tuple[str, ...]:
        return (self.column,)

    def check(self, table: pd.DataFrame) -> Check:
        x, miss = _numeric(_column(table, self.column))
        bad = np.isnan(x) & ~miss  # present but not a number
        with np.errstate(invalid="ignore"):
            if self.low is not None:
                bad |= x < self.low
            if self.high is not None:
                bad |= x > self.high
        return _everywhere(bad, miss)

    def label(self) -> str:
        lo = "-inf" if self.low is None else self.low
        hi = "+inf" if self.high is None else self.high
        return self.name or f"{lo} <= {self.column} <= {hi}"


@dataclass(frozen=True)
class Allowed:
    """``column`` takes one of ``values`` (categories, ordinal levels or a
    closed numeric set such as {0, 1, 2})."""

    column: str
    values: tuple[Any, ...]
    name: str | None = None
    kind = "allowed"

    def columns(self) -> tuple[str, ...]:
        return (self.column,)

    def check(self, table: pd.DataFrame) -> Check:
        v = _column(table, self.column)
        return _everywhere(~_isin(v, self.values), is_missing(v))

    def label(self) -> str:
        shown = list(self.values) if len(self.values) <= 12 else [*self.values[:12], "..."]
        return self.name or f"{self.column} in {shown}"


@dataclass(frozen=True)
class Grid:
    """``column`` lies on the lattice ``origin + k * step`` (k integer): the
    jumps of a discrete numeric variable (no 2.5 children)."""

    column: str
    step: float
    origin: float = 0.0
    name: str | None = None
    kind = "grid"

    def __post_init__(self) -> None:
        if not float(self.step) > 0:
            raise ValueError(f"grid step must be positive, got {self.step}")

    def columns(self) -> tuple[str, ...]:
        return (self.column,)

    def check(self, table: pd.DataFrame) -> Check:
        x, miss = _numeric(_column(table, self.column))
        with np.errstate(invalid="ignore"):
            k = (x - self.origin) / self.step
            off = np.abs(k - np.rint(k)) > 1e-6 * np.maximum(1.0, np.abs(k))
        return _everywhere(off | (np.isnan(x) & ~miss), miss)

    def label(self) -> str:
        at = f" from {self.origin}" if self.origin else ""
        return self.name or f"{self.column} on a grid of step {self.step}{at}"


_OPS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "<": operator.lt,
    "<=": lambda a, b: a <= b + _TOL * np.maximum(1.0, np.abs(b)),
    ">": operator.gt,
    ">=": lambda a, b: a >= b - _TOL * np.maximum(1.0, np.abs(b)),
    "==": lambda a, b: np.isclose(a, b, rtol=_TOL, atol=_TOL),
    "!=": lambda a, b: ~np.isclose(a, b, rtol=_TOL, atol=_TOL),
}


def _op(op: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if op not in _OPS:
        raise ValueError(f"operator must be one of {sorted(_OPS)}, got {op!r}")
    return _OPS[op]


@dataclass(frozen=True)
class Compare:
    """``left op right + offset``, with ``right`` a column or a number
    (``ap_lo <= ap_hi``, ``end_date >= start_date``)."""

    left: str
    op: str
    right: str | float
    offset: float = 0.0
    name: str | None = None
    kind = "compare"

    def __post_init__(self) -> None:
        _op(self.op)

    def columns(self) -> tuple[str, ...]:
        return (self.left, self.right) if isinstance(self.right, str) else (self.left,)

    def check(self, table: pd.DataFrame) -> Check:
        a, miss = _numeric(_column(table, self.left))
        bad = np.isnan(a) & ~miss
        if isinstance(self.right, str):
            b, miss_b = _numeric(_column(table, self.right))
            bad |= np.isnan(b) & ~miss_b
            miss = miss | miss_b
        else:
            b = np.full(len(a), float(self.right))
        with np.errstate(invalid="ignore"):
            bad |= ~_op(self.op)(a, b + self.offset) & ~np.isnan(a) & ~np.isnan(b)
        return _everywhere(bad, miss)

    def label(self) -> str:
        off = f" {'+' if self.offset >= 0 else '-'} {abs(self.offset)}" if self.offset else ""
        return self.name or f"{self.left} {self.op} {self.right}{off}"


@dataclass(frozen=True)
class Linear:
    """``sum_i a_i * column_i  op  bound`` (a budget, a mass balance)."""

    terms: tuple[tuple[str, float], ...]
    op: str
    bound: float
    name: str | None = None
    kind = "linear"

    def __post_init__(self) -> None:
        _op(self.op)
        if not self.terms:
            raise ValueError("a linear rule needs at least one term")

    def columns(self) -> tuple[str, ...]:
        return tuple(c for c, _ in self.terms)

    def check(self, table: pd.DataFrame) -> Check:
        total = np.zeros(len(table))
        miss = np.zeros(len(table), bool)
        unreadable = np.zeros(len(table), bool)
        for col, coef in self.terms:
            x, m = _numeric(_column(table, col))
            miss |= m
            unreadable |= np.isnan(x) & ~m
            total = total + float(coef) * np.nan_to_num(x)
        with np.errstate(invalid="ignore"):
            bad = ~_op(self.op)(total, np.full(len(total), float(self.bound)))
        return _everywhere(bad | unreadable, miss)

    def label(self) -> str:
        lhs = " + ".join(f"{c}" if a == 1 else f"{a}*{c}" for c, a in self.terms)
        return self.name or f"{lhs} {self.op} {self.bound}"


@dataclass(frozen=True)
class Absent:
    """``column`` must be missing (structural absence, e.g. the number of
    pregnancies of someone who cannot be pregnant, when declared as empty)."""

    column: str
    name: str | None = None
    kind = "absent"

    def columns(self) -> tuple[str, ...]:
        return (self.column,)

    def check(self, table: pd.DataFrame) -> Check:
        miss = is_missing(_column(table, self.column))
        return Check(np.ones(len(miss), bool), ~miss, np.zeros(len(miss), bool))

    def label(self) -> str:
        return self.name or f"{self.column} absent"


@dataclass(frozen=True)
class Present:
    """``column`` must not be missing."""

    column: str
    name: str | None = None
    kind = "present"

    def columns(self) -> tuple[str, ...]:
        return (self.column,)

    def check(self, table: pd.DataFrame) -> Check:
        miss = is_missing(_column(table, self.column))
        return Check(np.ones(len(miss), bool), miss, np.zeros(len(miss), bool))

    def label(self) -> str:
        return self.name or f"{self.column} present"


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "ratio": _ratio,
    "product": lambda a, b: a * b,
    "sum": lambda *xs: np.sum(xs, axis=0),
    "difference": lambda a, b: a - b,
}


def register_function(name: str, fn: Callable[..., np.ndarray]) -> None:
    """Make ``fn`` available to :class:`Derived` (and to the YAML) as ``name``."""
    FUNCTIONS[name] = fn


@dataclass(frozen=True)
class Derived:
    """``column == function(*inputs)`` within ``atol + rtol * |expected|``: a
    computed column (BMI = weight / height²).  A non-finite expected value
    (a division by zero) counts as missing."""

    column: str
    inputs: tuple[str, ...]
    function: str | Callable[..., np.ndarray] = "ratio"
    rtol: float = 1e-6
    atol: float = 0.0
    name: str | None = None
    kind = "derived"

    def columns(self) -> tuple[str, ...]:
        return (self.column, *self.inputs)

    def check(self, table: pd.DataFrame) -> Check:
        if isinstance(self.function, str) and self.function not in FUNCTIONS:
            raise ValueError(f"unknown function {self.function!r}; known: {sorted(FUNCTIONS)}")
        fn = FUNCTIONS[self.function] if isinstance(self.function, str) else self.function
        y, miss = _numeric(_column(table, self.column))
        expected = np.asarray(fn(*[_numeric(_column(table, c))[0] for c in self.inputs]), float)
        miss = miss | ~np.isfinite(expected)
        with np.errstate(invalid="ignore"):
            bad = ~(np.abs(y - expected) <= self.atol + self.rtol * np.abs(expected))
        return _everywhere(bad, miss)

    def label(self) -> str:
        fn = self.function if isinstance(self.function, str) else getattr(self.function, "__name__", "f")
        return self.name or f"{self.column} = {fn}({', '.join(self.inputs)})"


SimpleRule = Range | Allowed | Grid | Compare | Linear | Absent | Present | Derived


@dataclass(frozen=True)
class Conditional:
    """IF ``if_column`` in ``if_values`` THEN every rule of ``then`` holds."""

    if_column: str
    if_values: tuple[Any, ...]
    then: tuple[SimpleRule, ...]
    name: str | None = None
    kind = "conditional"

    def columns(self) -> tuple[str, ...]:
        return (self.if_column, *(c for r in self.then for c in r.columns()))

    def check_then(self, table: pd.DataFrame) -> Check:
        """The consequent alone, on every row."""
        n = len(table)
        viol, miss = np.zeros(n, bool), np.zeros(n, bool)
        for r in self.then:
            c = r.check(table)
            viol |= c.violated
            miss |= c.missing
        return Check(np.ones(n, bool), viol & ~miss, miss)

    def check(self, table: pd.DataFrame) -> Check:
        cond = _column(table, self.if_column)
        cmiss = is_missing(cond)
        fires = _isin(cond, self.if_values) & ~cmiss
        then = self.check_then(table)
        return Check(fires, fires & then.violated, fires & then.missing, cmiss)

    def label(self) -> str:
        return self.name or (
            f"if {self.if_column} in {list(self.if_values)} then "
            + " and ".join(r.label() for r in self.then)
        )


Rule = SimpleRule | Conditional


@dataclass(frozen=True)
class Outcome:
    """A rule's verdict on each row under the declared missing semantics:
    ``checked`` rows count in the rate, ``violated`` ones are its numerator."""

    checked: np.ndarray
    violated: np.ndarray


@dataclass(frozen=True)
class Constraints:
    """A set of rules and the declared meaning of a missing value.
    ``domains`` lists, per condition column, the values a missing condition
    may stand for under ``any_category`` (by default, the values named by
    the conditionals on that column)."""

    rules: tuple[Rule, ...]
    missing: str = "not_applicable"
    domains: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.missing not in MISSING_SEMANTICS:
            raise ValueError(f"missing must be one of {MISSING_SEMANTICS}, got {self.missing!r}")

    def outcome(self, rule: Rule, table: pd.DataFrame) -> Outcome:
        c = rule.check(table)
        if self.missing == "violates":
            missed = c.applies & c.missing
            if c.missing_condition is not None:
                missed = missed | c.missing_condition
            return Outcome(c.applies | missed, c.violated | missed)
        checked = c.applies & ~c.missing
        violated = c.violated.copy()
        if self.missing == "any_category" and isinstance(rule, Conditional) and c.missing_condition is not None:
            nowhere = self._any_category(rule, table, c.missing_condition)
            checked = checked | c.missing_condition
            violated = violated | nowhere
        return Outcome(checked, violated & checked)

    def _any_category(self, rule: Conditional, table: pd.DataFrame, cmiss: np.ndarray) -> np.ndarray:
        """Rows whose condition is missing and that no value of the condition
        column would make valid (charged to every conditional on that column)."""
        if not cmiss.any():
            return np.zeros(len(cmiss), bool)
        col = rule.if_column
        same = [r for r in self.rules if isinstance(r, Conditional) and r.if_column == col]
        domain = self.domains.get(col) or tuple(dict.fromkeys(v for r in same for v in r.if_values))
        sub = table.loc[cmiss].reset_index(drop=True)
        valid_somewhere = np.zeros(len(sub), bool)
        for value in domain:
            ok = np.ones(len(sub), bool)
            for r in same:
                if _isin(pd.Series([value], dtype=object), r.if_values)[0]:
                    ok &= ~r.check_then(sub).violated
            valid_somewhere |= ok
        out = np.zeros(len(cmiss), bool)
        out[cmiss] = ~valid_somewhere
        return out


# ----------------------------------------------------------- from dict / YAML
def _rule_from_dict(d: Mapping[str, Any], *, nested: bool = False) -> Rule:
    if not isinstance(d, Mapping) or len(d) != 1:
        raise ValueError(f"a rule is a one-key mapping such as {{'range': {{...}}}}, got {d!r}")
    ((kind, body),) = d.items()
    body = dict(body or {})
    if kind == "range":
        return Range(body["column"], body.get("low"), body.get("high"), body.get("name"))
    if kind == "allowed":
        return Allowed(body["column"], tuple(body["values"]), body.get("name"))
    if kind == "grid":
        return Grid(body["column"], float(body["step"]), float(body.get("origin", 0.0)), body.get("name"))
    if kind == "compare":
        right = body["right"]
        right = right if isinstance(right, str) else float(right)
        return Compare(body["left"], body["op"], right, float(body.get("offset", 0.0)), body.get("name"))
    if kind == "linear":
        terms = body["terms"]
        terms = tuple(terms.items()) if isinstance(terms, Mapping) else tuple(tuple(t) for t in terms)
        return Linear(tuple((str(c), float(a)) for c, a in terms), body["op"], float(body["bound"]), body.get("name"))
    if kind == "absent":
        return Absent(body["column"], body.get("name"))
    if kind == "present":
        return Present(body["column"], body.get("name"))
    if kind == "derived":
        return Derived(
            body["column"],
            tuple(body["inputs"]),
            body.get("function", "ratio"),
            float(body.get("rtol", 1e-6)),
            float(body.get("atol", 0.0)),
            body.get("name"),
        )
    if kind == "conditional":
        if nested:
            raise ValueError("nested conditionals are not supported")
        then = tuple(_rule_from_dict(r, nested=True) for r in body["then"])
        return Conditional(body["if_column"], tuple(body["if_values"]), then, body.get("name"))
    raise ValueError(
        f"unknown rule {kind!r}; known: range, allowed, grid, compare, linear, absent, present, "
        "derived, conditional"
    )


def rules_from_list(items: Iterable[Mapping[str, Any]]) -> tuple[Rule, ...]:
    return tuple(_rule_from_dict(r) for r in items or ())
