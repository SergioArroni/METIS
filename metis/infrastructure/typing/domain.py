"""The observed numeric domain of a column: which values it takes and how regularly they spread.

This module only MEASURES what is in the data. It decides no type and knows no threshold:
deciding is :mod:`metis.infrastructure.typing.classification`.

The central measurement is **regularity**: the fraction of gaps between contiguous values that
equal the most frequent gap. It separates a real lattice (a height in cm: gaps of 1) from one
that only exists in arithmetic (1, 10, 100, 1000: the gcd of the distances is 9, yet there is no
value at 19).

Caveat, measured in Hefesto (critique ``2026-09-22-tipificacion-dos-ejes``): regularity **grows
with sample size** even when the domain is continuous, because it measures how densely the
recording grid has been sampled, not whether that grid is the domain.

Ported from Hefesto's Phase -1 (``hefesto/core/typing/dominio.py``); decisions are identical.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from math import exp, gcd, log
from statistics import median

# Two values do not define a lattice, they define a segment, and every segment is regular. The
# minimum for "are they regularly spread?" to mean anything is three.
_MIN_VALUES_FOR_REGULARITY = 3

# Log-scale gaps are irrational, so they cannot be compared for equality like the linear ones
# (compared exactly, in Decimal). Two gaps count as the same if their relative difference is
# below this.
_LOG_TOLERANCE = 1e-6

# Core percentiles: the range used to count reachable values comes from here, not from the raw
# extremes. They are percentiles of the ROWS, not of the distinct values: row percentiles
# converge to the variable's as the sample grows, distinct-value percentiles do not (counted on
# the distinct values, k_j grew from 469 to 752 between 300 and 100,000 rows of the same
# Gaussian). Below this number of distinct values nothing is trimmed: trimming removes tails,
# and a handful of values has no tails, only data.
_CORE_PERCENTILE = 0.01
_MIN_VALUES_TO_TRIM = 20


class NonNumericDomainError(ValueError):
    """The numeric domain was requested for a column whose values are not numbers."""


@dataclass(frozen=True)
class Domain:
    """The values a column takes, with what was measured on them.

    ``values`` are the distinct observed values, one per NUMERIC value and in their raw
    representation: '3.50' is kept as is (it tells the recording precision), but '1.0' and
    '1.00' are the same value and only one enters. Deduplicating by text left zero gaps between
    them, and with them a zero ``step``.
    """

    values: tuple[str, ...]
    minimum: Decimal
    maximum: Decimal
    core_minimum: Decimal
    core_maximum: Decimal
    step: Decimal
    regularity: float
    log_regularity: float
    grid: Decimal = Decimal(1)

    @property
    def n_distinct(self) -> int:
        return len(self.values)

    @property
    def range(self) -> Decimal:
        return self.maximum - self.minimum

    @property
    def reachable_on_lattice(self) -> Decimal:
        """How many values fit at ``step`` gaps inside the CORE of the domain.

        On the raw range a single outlier inflates it and turns a stepped domain into an
        apparently continuous one (integers 0-50 give 51 reachable values; adding 99999 gives
        100000). A large gap argues for discreteness, never against it, and counting on the core
        prevents it from counting the other way. Only meaningful if the lattice is real: check
        ``regularity`` first.
        """
        return (self.core_maximum - self.core_minimum) / self.step + 1

    @property
    def registration_cells(self) -> Decimal:
        """How many values fit inside the core at :attr:`grid` gaps: the coarsest grid that
        contains every observed value, not the one formed by contiguous gaps.

        This is the count used when the lattice has not been seen. The modal gap is only the
        grid if most gaps agree with it; with few rows the gaps all differ and the mode is
        arbitrary. An earlier version counted at :attr:`precision` (mode of the decimals), which
        is only right when the true step is a power of ten (a grid of 5s recorded as integers has
        precision 1).
        """
        return (self.core_maximum - self.core_minimum) / self.grid + 1

    @property
    def precision(self) -> Decimal:
        """How finely the column is recorded, in the TYPICAL case: 1 for integers, 0.1 for one
        decimal, 0.01 for two.

        Counted on the NORMALISED value, not the text (``adult_test.csv`` writes integers as
        '226802.0'), and as the MODE of the decimals, not the maximum (in airbnb ``latitude`` a
        few values carry 15 decimals of floating-point noise while most carry 5). It is the
        recording resolution, not the domain's step, so it decides no type.
        """
        decimals = Counter(
            max(0, -int(Decimal(v).normalize().as_tuple().exponent)) for v in self.values
        )
        return Decimal(1).scaleb(-decimals.most_common(1)[0][0]) if decimals else Decimal(1)


def is_numeric(values: list[str]) -> bool:
    """Whether every non-empty value parses as a finite number.

    ``Decimal`` accepts 'nan', 'Infinity' and 'sNaN', which cannot be used to measure a domain.
    """
    present = [v for v in values if v != ""]
    if not present:
        return False
    try:
        return all(Decimal(value).is_finite() for value in present)
    except InvalidOperation:
        return False


def _distinct_by_value(present: list[str]) -> list[str]:
    """One representative per numeric value, in order.

    Deduplicates first by text (cheap, drops most rows) and then by value (the only way to tell
    that '1.0' and '1.00' are the same number). ``setdefault`` keeps the first representation.
    """
    by_value: dict[Decimal, str] = {}
    for raw in sorted(set(present)):
        by_value.setdefault(Decimal(raw), raw)
    return [by_value[key] for key in sorted(by_value)]


def _core(ordered: list[Decimal], rows: dict[Decimal, int]) -> tuple[Decimal, Decimal]:
    """Domain extremes without their tails, in percentiles of the rows (``rows`` holds each
    value's count). No trimming with few distinct values, nor if trimming would remove
    everything."""
    if len(ordered) < _MIN_VALUES_TO_TRIM:
        return ordered[0], ordered[-1]
    last = sum(rows.values()) - 1
    positions = (int(_CORE_PERCENTILE * last), int((1 - _CORE_PERCENTILE) * last))
    extremes: list[Decimal] = []
    accumulated = 0
    for value in ordered:
        accumulated += rows[value]
        while len(extremes) < 2 and positions[len(extremes)] < accumulated:
            extremes.append(value)
        if len(extremes) == 2:
            break
    low, high = extremes
    return (low, high) if low < high else (ordered[0], ordered[-1])


def _common_grid(ordered: list[Decimal]) -> Decimal:
    """The largest step ``g`` such that every value lies on ``min + g * i``: the gcd of the
    distances to the minimum, in exact arithmetic.

    Rejected as a lattice estimator (1, 10, 100, 1000 give 9), it is used here only to count
    cells, and only where regularity said the lattice of gaps is not visible. It only decreases
    as values are added, and on a real lattice settles on its step with very few values. Known
    limit: a gapped support that is regular on no scale (1, 2, 5, 10...) yields a fine grid with
    millions of cells and is continuised. A single off-lattice value collapses it to its
    resolution.
    """
    if len(ordered) < 2:
        return Decimal(1)
    # Values are finite (``is_numeric``), so the exponent is always an int.
    exponent: int = min(0, *(int(v.normalize().as_tuple().exponent) for v in ordered))
    scale = Decimal(1).scaleb(exponent)
    base = ordered[0]
    common = 0
    for value in ordered[1:]:
        common = gcd(common, int((value - base).scaleb(-exponent)))
        if common == 1:
            break
    return scale * common if common else Decimal(1)


def _linear_regularity(ordered: list[Decimal]) -> tuple[float, Decimal]:
    """Fraction of contiguous gaps equal to the most frequent gap, and that gap.

    The modal gap and not the gcd: the gcd always returns a number, lattice or not. The mode is
    only reliable when most gaps agree with it, which is what the first element measures.
    """
    if len(ordered) < _MIN_VALUES_FOR_REGULARITY:
        gap = ordered[-1] - ordered[0] if len(ordered) == 2 else Decimal(1)
        return 1.0, gap or Decimal(1)
    gaps = [b - a for a, b in pairwise(ordered)]
    modal, repetitions = Counter(gaps).most_common(1)[0]
    return repetitions / len(gaps), modal


def _log_gaps(ordered: list[Decimal]) -> list[float]:
    logs = [log(float(v)) for v in ordered]
    return [b - a for a, b in pairwise(logs)]


def _log_regularity(ordered: list[Decimal]) -> float:
    """The same, measured on the logarithm of the values: reveals a domain that jumps in orders
    of magnitude (1, 10, 100...), where the lattice exists but not on a linear scale.

    Zero if any value is not positive: the logarithm is undefined there, and 0 claims nothing.
    """
    if len(ordered) < _MIN_VALUES_FOR_REGULARITY or ordered[0] <= 0:
        return 0.0
    gaps = _log_gaps(ordered)
    reference = median(gaps)
    if reference <= 0:
        return 0.0
    close = sum(1 for gap in gaps if abs(gap - reference) <= _LOG_TOLERANCE * reference)
    return close / len(gaps)


def log_ratio(domain: Domain) -> float | None:
    """The common ratio of a log lattice: ``exp`` of the median log-gap between contiguous
    values, or ``None`` when the log scale is undefined (non-positive values, < 3 values).

    Not part of Hefesto's measurement (its generator rebuilds the grid with
    ``geomspace(min, max, n_observed)``); exposed so the schema can carry ``log_step``.
    """
    ordered = [Decimal(v) for v in domain.values]
    if len(ordered) < _MIN_VALUES_FOR_REGULARITY or ordered[0] <= 0:
        return None
    reference = median(_log_gaps(ordered))
    return exp(reference) if reference > 0 else None


def measure(values: list[str], frequencies: Mapping[str, int] | None = None) -> Domain:
    """The domain deduced from the observed values. Raises :class:`NonNumericDomainError` if
    the column is not numeric.

    ``values`` are the rows. A caller that already counted them can pass the distinct values and
    their ``frequencies``: the core is taken in percentiles of the ROWS, and passing only the
    distinct values without frequencies made the type depend on n.
    """
    present = [v for v in values if v != ""]
    if not is_numeric(present):
        raise NonNumericDomainError(
            "the numeric domain requires finite numeric values in the whole column"
        )
    by_text = (
        Counter({v: frequencies[v] for v in present})
        if frequencies is not None
        else Counter(present)
    )
    distinct = _distinct_by_value(list(by_text))
    ordered = [Decimal(v) for v in distinct]
    rows: Counter[Decimal] = Counter()
    for raw, times in by_text.items():
        rows[Decimal(raw)] += times
    regularity, step = _linear_regularity(ordered)
    core_minimum, core_maximum = _core(ordered, rows)
    return Domain(
        grid=_common_grid(ordered),
        values=tuple(distinct),
        minimum=ordered[0],
        maximum=ordered[-1],
        core_minimum=core_minimum,
        core_maximum=core_maximum,
        step=step,
        regularity=regularity,
        log_regularity=_log_regularity(ordered),
    )
