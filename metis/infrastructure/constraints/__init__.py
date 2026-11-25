"""Domain rules and physical laws: the transversal axis of METIS.

They are checked on the synthetic table (and on the real one), reported per
rule as the share of rows that break them, and kept out of the scores.
"""

from .rules import (
    FUNCTIONS,
    MISSING_SEMANTICS,
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
    register_function,
    rules_from_list,
)
from .validation import build_constraints, rules_from_schema, validate, wilson_interval

__all__ = [
    "FUNCTIONS",
    "MISSING_SEMANTICS",
    "Absent",
    "Allowed",
    "Compare",
    "Conditional",
    "Constraints",
    "Derived",
    "Grid",
    "Linear",
    "Present",
    "Range",
    "build_constraints",
    "register_function",
    "rules_from_list",
    "rules_from_schema",
    "validate",
    "wilson_interval",
]
