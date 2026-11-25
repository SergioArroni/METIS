"""The four variable types of Hefesto's Phase −1 as METIS's schema (2026-10-19).

nominal, ordinal, continuous (dates with ``datetime: true``) and discrete (a
closed set of ``levels`` or a lattice of ``step``), plus ``id``; ``schema:
auto`` infers them with the Phase −1 rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.domain.contracts import TypeSchema
from metis.infrastructure.preprocess.caster import SimpleCaster
from metis.interface.sdk import Evaluator


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("categorical", "nominal"),
        ("boolean", "nominal"),
        ("text", "nominal"),
        ("geospatial", "continuous"),
        ("datetime", "continuous"),
    ],
)
def test_old_types_are_rejected_with_the_new_name(old, new):
    with pytest.raises(ValueError, match=new):
        TypeSchema(columns={"c": old})


def test_ranges_no_longer_exist():
    with pytest.raises(ValueError, match="levels"):
        TypeSchema(columns={"c": {"type": "discrete", "ranges": [[0, 1]]}})


def _cast(spec, real, synth):
    caster = SimpleCaster(TypeSchema(columns={"k": spec})).fit(pd.DataFrame({"k": real}))
    return caster.transform(pd.DataFrame({"k": synth}))


@pytest.mark.parametrize(
    "spec",
    [{"type": "discrete", "levels": [0, 1, 2, 3]}, {"type": "discrete", "step": 0.5}, "discrete"],
)
def test_a_discrete_is_read_by_its_value_alone(spec):
    """Its domain is a rule of the constraints axis (2026-10-20), not a shadow column."""
    cat, num = _cast(spec, [0, 1, 2, 3], [0, 2.5, 3.0, None])
    assert num["k"].tolist()[:3] == [0.0, 2.5, 3.0] and pd.isna(num["k"].iloc[3])
    assert not any(str(c).startswith("k") for c in cat.columns)


def _children(n: int, seed: int, off_grid: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kids = rng.integers(0, 5, n).astype(float)
    if off_grid:
        kids = kids + rng.choice([0.0, 0.5], n)
    return pd.DataFrame({"kids": kids, "c": rng.choice(list("abc"), n)})


def test_impossible_values_are_a_broken_rule_not_a_second_fidelity_penalty():
    real, holdout = _children(400, 0), _children(400, 1)
    bad = _children(400, 3, off_grid=True)
    schema = {"kids": {"type": "discrete", "levels": [0, 1, 2, 3, 4]}, "c": "nominal"}
    metrics = ["fidelity.tvd", "fidelity.ks", "privacy.dcr"]
    with_rules = {"data": {"schema": schema}, "metrics": metrics}
    without = {"data": {"schema": schema, "constraints": {"from_schema": False}}, "metrics": metrics}
    ev = Evaluator()
    a = ev.evaluate(real, bad, with_rules, seed=0, holdout=holdout)
    b = ev.evaluate(real, bad, without, seed=0, holdout=holdout)
    # the declared levels are a rule: about half the children are impossible (2.5, …)
    (rule,) = a.artifacts["constraints"]["rules"]
    assert rule["origin"] == "schema" and not rule["ok"]
    assert 0.4 < rule["synthetic"]["violation_rate"] < 0.6
    # and the scores do not depend on whether the rule is checked
    np.testing.assert_equal([r.value for r in a.results], [r.value for r in b.results])
    assert all(np.isfinite(r.value) for r in a.results)
    assert not any("__lvl" in str(k) for r in a.results for k in (r.details or {}))


def test_schema_auto_infers_the_phase_minus_one_types():
    rng = np.random.default_rng(0)
    n = 300
    real = pd.DataFrame(
        {
            "id": range(n),
            "x": rng.normal(size=n),
            "kids": rng.integers(0, 15, n),
            "c": rng.choice(list("ab"), n),
        }
    )
    cfg = {"data": {"schema": "auto"}, "metrics": ["fidelity.ks", "fidelity.tvd"]}
    summary = Evaluator().evaluate(real, real.sample(frac=1, random_state=1), cfg, seed=0)
    inferred = summary.artifacts["inferred_schema"]["schema"]
    assert inferred["id"]["type"] == "id"
    assert inferred["x"]["type"] == "continuous"
    assert inferred["kids"]["type"] == "discrete"
    assert inferred["c"]["type"] == "nominal"
