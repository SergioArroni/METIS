"""The rule that types a numeric column from its data (manuscript Rules III-IV): the only place
that chooses between continuous and discrete numeric.

The decision uses TWO axes: first whether the lattice really exists, then how many values fit
in it. Continuising is not an improvement but a concession to size: it is paid when listing the
domain is not worth it, so a short domain is enumerated even if its range is huge.

Ported from Hefesto's ``hefesto/core/typing/clasificacion.py`` (static path only; the
user-declared path is not ported). Decisions are identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from metis.infrastructure.typing.domain import Domain, log_ratio
from metis.infrastructure.typing.thresholds import InferenceThresholds

CONTINUOUS = "continuous"
DISCRETE = "discrete"

# How a discrete column's domain is walked.
MODE_STEP = "step"  # a uniform lattice of step delta
MODE_LOG = "log"  # a lattice that is regular in orders of magnitude
MODE_ENUMERATED = "enumerated"  # a closed list of observed values


@dataclass(frozen=True)
class Verdict:
    """The type of a column, with the evidence behind it.

    ``undecided`` marks the band where regularity is neither clearly a lattice nor clearly not.
    """

    type: str
    evidence: str
    undecided: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)


def bounds_of(minimum: Decimal, maximum: Decimal) -> dict[str, float]:
    """Observed physical bounds that accompany the type."""
    return {"lower_bound": float(minimum), "upper_bound": float(maximum)}


def _evidence(domain: Domain) -> str:
    return (
        f"observed domain {domain.minimum:g}-{domain.maximum:g}, "
        f"{domain.n_distinct} distinct value(s), typical gap {domain.step:g}, "
        f"regularity {domain.regularity:.2f} linear / {domain.log_regularity:.2f} log"
    )


def _enumerable(domain: Domain, thresholds: InferenceThresholds) -> Verdict:
    """Short enough to write whole: it is enumerated, and no unseen value can be emitted."""
    return Verdict(
        type=DISCRETE,
        evidence=(
            f"{_evidence(domain)}: {domain.n_distinct} <= enumerable={thresholds.enumerable}, "
            "the domain is enumerated"
        ),
        parameters={**bounds_of(domain.minimum, domain.maximum), "mode": MODE_ENUMERATED},
    )


def _linear_lattice(domain: Domain, thresholds: InferenceThresholds) -> Verdict:
    """Values spread at regular gaps: counting how many fit decides whether the lattice is so
    fine it does not matter (continuous) or coarse enough to be a stepped domain (discrete)."""
    reachable = domain.reachable_on_lattice
    evidence = f"{_evidence(domain)}; {reachable:.0f} reachable values"
    bounds = bounds_of(domain.minimum, domain.maximum)
    if reachable >= thresholds.continuization:
        return Verdict(
            type=CONTINUOUS,
            evidence=(
                f"{evidence} (>= continuization={thresholds.continuization}): the lattice is "
                "so fine relative to the range that recording rounding is irrelevant"
            ),
            parameters={**bounds, "precision": float(domain.precision)},
        )
    return Verdict(
        type=DISCRETE,
        evidence=(
            f"{evidence} (< continuization={thresholds.continuization}): uniform, coarse "
            "lattice; the domain is stepped"
        ),
        parameters={**bounds, "mode": MODE_STEP, "step": float(domain.step)},
    )


def _log_lattice(domain: Domain) -> Verdict:
    """Gaps are irregular on the linear scale but regular on the log scale: the domain grows in
    orders of magnitude. Continuising it linearly would put mass in the holes."""
    return Verdict(
        type=DISCRETE,
        evidence=(
            f"{_evidence(domain)}: no lattice on the linear scale but one on the log scale, so "
            "the domain is modelled in orders of magnitude"
        ),
        parameters={
            **bounds_of(domain.minimum, domain.maximum),
            "mode": MODE_LOG,
            "log_step": log_ratio(domain),
        },
    )


def _by_registration_grid(domain: Domain, thresholds: InferenceThresholds, reason: str) -> Verdict:
    """The lattice of gaps is not visible, so count on the grid common to all observed values:
    if it is so fine relative to the core that rounding is irrelevant, continuise; otherwise the
    domain has steps of that grid's size.

    Earlier, not seeing the lattice gave continuous without counting; with few rows gaps rarely
    repeat, so the type changed with n (a Gaussian with one decimal was continuous at 100 rows
    and discrete at 300). Irregular gaps do not mean there is no grid, only that it is not yet
    filled. Known limit: a gapped support regular on no scale (1, 2, 5, 10, 20, 50...) has
    common grid 1 and millions of cells, and is continuised here.
    """
    cells = domain.registration_cells
    bounds = bounds_of(domain.minimum, domain.maximum)
    evidence = (
        f"{_evidence(domain)}: {reason}; {cells:.0f} values fit in the core at the common grid "
        f"{domain.grid:g}"
    )
    if cells >= thresholds.continuization:
        return Verdict(
            type=CONTINUOUS,
            evidence=(
                f"{evidence} (>= continuization={thresholds.continuization}): the recording "
                "grid is so fine relative to the range that it is irrelevant"
            ),
            parameters={**bounds, "precision": float(domain.precision)},
        )
    return Verdict(
        type=DISCRETE,
        evidence=(
            f"{evidence} (< continuization={thresholds.continuization}): the recording grid "
            "is coarse, the domain is stepped"
        ),
        parameters={**bounds, "mode": MODE_STEP, "step": float(domain.grid)},
    )


def deciding_cells(domain: Domain, thresholds: InferenceThresholds) -> Decimal | None:
    """The count that :func:`classify` compares with ``continuization`` for this column, or
    ``None`` if the deciding branch compares none (enumerable and log branches)."""
    if domain.n_distinct <= thresholds.enumerable:
        return None
    if domain.regularity >= thresholds.lattice:
        return domain.reachable_on_lattice
    if domain.log_regularity >= thresholds.lattice:
        return None
    return domain.registration_cells


def classify(domain: Domain, thresholds: InferenceThresholds) -> Verdict:
    """The type inferred from the data when nobody can be asked (static mode).

    Marks ``undecided`` in the band where regularity does not decide, but never leaves the
    column untyped. Outside the enumerable and log branches, continuous vs discrete is always
    decided by the same criterion -- how many values fit in the core --; regularity only chooses
    the grid they are counted on: the gaps' if the lattice is visible, the common grid if not.
    """
    if domain.n_distinct <= thresholds.enumerable:
        return _enumerable(domain, thresholds)
    if domain.regularity >= thresholds.lattice:
        return _linear_lattice(domain, thresholds)
    if domain.log_regularity >= thresholds.lattice:
        return _log_lattice(domain)
    if domain.regularity <= thresholds.no_lattice:
        return _by_registration_grid(domain, thresholds, "gaps are not regular on the linear scale")

    # The middle band: regularity says neither that there is a lattice nor that there is not.
    # A fixed "discrete" here made the verdict depend on the number of ROWS (the same Gaussian
    # with two decimals was continuous at 1,000 rows, discrete at 3,000, continuous at 10,000),
    # so it is decided as in the no-lattice branch and only flagged as undecided.
    by_grid = _by_registration_grid(
        domain, thresholds, "regularity does not decide whether there is a lattice"
    )
    return replace(
        by_grid,
        undecided=True,
        evidence=(
            f"regularity falls between no_lattice={thresholds.no_lattice} and "
            f"lattice={thresholds.lattice}, so it is not visible whether the domain is stepped "
            "or a continuous variable rounded on recording; decided by the fineness of the "
            f"grid: {by_grid.evidence}"
        ),
    )
