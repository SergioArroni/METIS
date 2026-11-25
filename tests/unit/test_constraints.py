"""Domain rules, the transversal axis (2026-10-20).

Declared rules (from the schema or ``data.constraints``) are checked row by
row on the synthetic table; each one reports the share of rows that break it
and a verdict, and none of them changes a score.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from metis.domain.errors import ConfigError
from metis.infrastructure.constraints import (
    Absent,
    Allowed,
    Compare,
    Conditional,
    Constraints,
    Derived,
    Grid,
    Linear,
    Present,
    Range,
    rules_from_list,
    rules_from_schema,
    validate,
    wilson_interval,
)
from metis.interface.sdk import Evaluator


def _rate(rule, table, missing="not_applicable"):
    o = Constraints((rule,), missing).outcome(rule, pd.DataFrame(table))
    return int(o.violated.sum()), int(o.checked.sum())


def test_grid_flags_values_between_the_jumps():
    t = {"hijos": [0, 1, 2.5, 3, np.nan]}
    assert _rate(Grid("hijos", 1.0), t) == (1, 4)  # 2.5; the gap does not count
    assert _rate(Grid("hijos", 1.0), t, "violates") == (2, 5)  # the gap is a violation
    assert _rate(Grid("x", 0.5, origin=0.25), {"x": [0.25, 0.75, 1.0]}) == (1, 3)
    assert _rate(Grid("x", 0.1), {"x": [0.1, 0.2, 0.30000000000000004]}) == (0, 3)


def test_allowed_matches_the_same_value_however_it_is_written():
    t = {"nivel": ["1", 2, 3.0, " 3 ", "4", None]}
    assert _rate(Allowed("nivel", (1, 2, 3)), t) == (1, 5)  # only "4"
    assert _rate(Allowed("c", ("Male", "Female")), {"c": ["male", "FEMALE", "other"]}) == (1, 3)


def test_range_and_unreadable_values():
    t = {"edad": [-1, 0, 50, 120, 121, "abc", None]}
    assert _rate(Range("edad", 0, 120), t) == (3, 6)  # -1, 121 and the unreadable "abc"
    assert _rate(Range("edad", low=0), t) == (2, 6)


def test_compare_between_columns_and_with_a_constant():
    t = {"ap_lo": [80, 90, 130, None], "ap_hi": [120, 90, 110, 100]}
    assert _rate(Compare("ap_lo", "<=", "ap_hi"), t) == (1, 3)
    assert _rate(Compare("ap_lo", "<", "ap_hi"), t) == (2, 3)
    assert _rate(Compare("ap_hi", ">=", 100.0), t) == (1, 4)
    assert _rate(Compare("ap_lo", "<=", "ap_hi", offset=-30), t) == (2, 3)  # lo <= hi - 30
    with pytest.raises(ValueError, match="operator"):
        Compare("a", "=<", "b")


def test_linear_rule():
    t = {"a": [1, 2, 3], "b": [1, 1, 1]}
    assert _rate(Linear((("a", 1.0), ("b", 2.0)), "<=", 4.0), t) == (1, 3)  # 3 + 2 > 4


def test_derived_rule():
    t = {"peso": [70, 80, 90], "altura": [1.75, 2.0, 0.0], "ratio": [40.0, 41.0, 5.0]}
    r = Derived("ratio", ("peso", "altura"), "ratio", rtol=1e-3)
    assert _rate(r, t) == (1, 2)  # 80/2 = 40 != 41; a division by zero is not checked


def test_absent_and_present():
    t = {"x": [1, None, 3]}
    assert _rate(Absent("x"), t) == (2, 3)
    assert _rate(Present("x"), t) == (1, 3)


def test_conditional_under_the_three_missing_semantics():
    then = (Allowed("embarazos", (0,)),)
    rule = Conditional("sexo", ("hombre",), then)
    t = {"sexo": ["hombre", "hombre", "mujer", None, None], "embarazos": [0, 2, 3, 1, 0]}
    # it applies to men only: one of the two breaks it
    assert _rate(rule, t) == (1, 2)
    # a missing sex counts as a violation
    assert _rate(rule, t, "violates") == (3, 4)
    # a missing sex is fine if some sex would make the row valid: with a rule
    # on men only, «mujer» is not in its domain, so a row with pregnancies is
    # valid only if some declared value accepts it -> the row with 1 breaks it
    assert _rate(rule, t, "any_category") == (2, 4)
    wide = Constraints((rule,), "any_category", {"sexo": ("hombre", "mujer")})
    o = wide.outcome(rule, pd.DataFrame(t))
    assert int(o.violated.sum()) == 1  # with «mujer» possible, the missing-sex rows are fine


def test_rules_from_the_yaml_mapping():
    rules = rules_from_list(
        [
            {"range": {"column": "edad", "low": 0, "high": 120}},
            {"allowed": {"column": "c", "values": ["a", "b"]}},
            {"grid": {"column": "hijos", "step": 1}},
            {"compare": {"left": "ap_lo", "op": "<=", "right": "ap_hi"}},
            {"linear": {"terms": {"a": 1, "b": 2}, "op": "<=", "bound": 4}},
            {"derived": {"column": "imc", "inputs": ["peso", "altura2"]}},
            {"absent": {"column": "x"}},
            {"present": {"column": "y"}},
            {
                "conditional": {
                    "if_column": "sexo",
                    "if_values": ["hombre"],
                    "then": [{"allowed": {"column": "embarazos", "values": [0]}}],
                }
            },
        ]
    )
    assert [r.kind for r in rules] == [
        "range", "allowed", "grid", "compare", "linear", "derived", "absent", "present", "conditional",
    ]
    with pytest.raises(ValueError, match="nested"):
        rules_from_list([{"conditional": {"if_column": "a", "if_values": [1], "then": [
            {"conditional": {"if_column": "b", "if_values": [1], "then": []}}]}}])
    with pytest.raises(ValueError, match="unknown rule"):
        rules_from_list([{"between": {}}])


def test_the_schema_declares_rules_without_reading_the_data():
    rules = rules_from_schema({
        "nivel": {"type": "ordinal", "levels": ["bajo", "medio", "alto"]},
        "hijos": {"type": "discrete", "step": 1},
        "dado": {"type": "discrete", "levels": [1, 2, 3, 4, 5, 6]},
        "peso": "continuous",
        "color": "nominal",
        "talla": {"type": "discrete"},  # neither levels nor step: nothing declared, no rule
    })
    assert {(r.kind, r.column) for r in rules} == {
        ("allowed", "nivel"), ("grid", "hijos"), ("allowed", "dado"),
    }


def test_wilson_interval():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and 0.03 < hi < 0.04
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def _config(rules=None, **extra):
    schema = {
        "hijos": {"type": "discrete", "step": 1},
        "nivel": {"type": "ordinal", "levels": ["bajo", "medio", "alto"]},
        "ap_lo": "continuous",
        "ap_hi": "continuous",
    }
    constraints = {"rules": rules or [], **extra}
    return {"data": {"schema": schema, "constraints": constraints}, "metrics": ["fidelity.ks"]}


def _tables(n=200, seed=0):
    rng = np.random.default_rng(seed)
    lo = rng.normal(80, 5, n)
    real = pd.DataFrame({
        "hijos": rng.integers(0, 4, n).astype(float),
        "nivel": rng.choice(["bajo", "medio", "alto"], n),
        "ap_lo": lo,
        "ap_hi": lo + rng.uniform(20, 40, n),
    })
    synth = real.sample(n, replace=True, random_state=seed + 1).reset_index(drop=True)
    synth.loc[: n // 10 - 1, "hijos"] += 0.5  # 10 % between the jumps
    synth.loc[n // 10 : n // 10 + 4, "nivel"] = "altísimo"  # 5 invented levels
    return real, synth


def test_report_per_rule_with_shares_verdicts_and_examples():
    real, synth = _tables()
    report, warnings = validate(real, synth, _config([{"compare": {"left": "ap_lo", "op": "<=", "right": "ap_hi"}}]))
    by = {r["rule"]: r for r in report["rules"]}
    grid = by["hijos: step 1"]
    assert grid["origin"] == "schema" and not grid["ok"]
    assert grid["synthetic"]["violations"] == 20 and grid["synthetic"]["violation_rate"] == pytest.approx(0.1)
    assert grid["compliance"] == pytest.approx(0.9)
    lo, hi = grid["synthetic"]["violation_rate_ci95"]
    assert lo < 0.1 < hi
    assert grid["real"]["violations"] == 0
    assert [e["row"] for e in grid["examples"]] == [0, 1, 2, 3, 4]
    assert all(e["hijos"] % 1 == 0.5 for e in grid["examples"])
    nivel = by["nivel: declared ordinal levels"]
    assert nivel["synthetic"]["violations"] == 5 and not nivel["ok"]
    cmp_ = by["ap_lo <= ap_hi"]
    assert cmp_["ok"] and cmp_["origin"] == "user" and cmp_["synthetic"]["violations"] == 0
    assert report["n_rules"] == 3 and report["n_failed"] == 2 and not report["ok"]
    assert report["synthetic_rows_violating_any"] == 25
    assert sum("constraint violated" in w for w in warnings) == 2


def test_tolerance_and_dirty_real_data():
    real, synth = _tables()
    real.loc[0, "ap_hi"] = real.loc[0, "ap_lo"] - 1  # the real table breaks the rule once
    rule = [{"compare": {"left": "ap_lo", "op": "<=", "right": "ap_hi"}}]
    report, warnings = validate(real, synth, _config(rule, tolerance=0.2))
    by = {r["rule"]: r for r in report["rules"]}
    assert by["hijos: step 1"]["ok"]  # 10 % <= 20 %
    assert by["ap_lo <= ap_hi"]["real"]["violations"] == 1
    assert any("the real table also breaks" in w for w in warnings)


def test_nothing_declared_means_no_report():
    real, synth = _tables()
    cfg = _config(from_schema=False)
    assert validate(real, synth, cfg) == (None, [])


@pytest.mark.parametrize(
    ("constraints", "match"),
    [
        ({"rules": [{"compare": {"left": "nope", "op": "<=", "right": "ap_hi"}}]}, "not in the synthetic"),
        ({"missing": "maybe"}, "missing"),
        ({"tolerance": 1.5}, "tolerance"),
        ({"rule": []}, "unknown key"),
        ({"rules": [{"compare": {"left": "ap_lo", "op": "~", "right": "ap_hi"}}]}, "operator"),
    ],
)
def test_malformed_declarations_fail_early(constraints, match):
    real, synth = _tables()
    cfg = _config()
    cfg["data"]["constraints"] = constraints
    with pytest.raises(ConfigError, match=match):
        validate(real, synth, cfg)


def test_the_evaluation_goes_on_and_no_score_changes():
    real, synth = _tables(n=300)
    holdout = _tables(n=300, seed=5)[0]
    with_rules = _config([{"compare": {"left": "ap_lo", "op": "<=", "right": "ap_hi"}}])
    without = _config(from_schema=False)
    ev = Evaluator()
    a = ev.evaluate(real, synth, with_rules, seed=0, holdout=holdout)
    b = ev.evaluate(real, synth, without, seed=0, holdout=holdout)
    assert a.artifacts["constraints"]["n_failed"] == 2
    assert "constraints" not in b.artifacts
    scores = lambda s: {k: v for k, v in s.aggregates.items() if isinstance(v, float)}  # noqa: E731
    assert scores(a) == scores(b)
    assert any("constraint violated" in w for w in a.artifacts["warnings"])
