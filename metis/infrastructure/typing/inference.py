"""Static column-type inference (Hefesto's Phase -1, static mode), producing a METIS schema.

Given a table, it sets aside what is not one of the four modelable tabular variables
(identifier, constant, timestamp, free text, non-tabular blob) and types the rest as nominal,
continuous or discrete numeric. Nothing is asked: every doubt is closed with its conservative
value and written in the column's notes, so the result is auditable without being interactive.

Mapping to METIS types:

- identifier candidate (Q0), non-tabular blob, empty column -> ``id`` (excluded from evaluation);
- categorical, constant, free text, boolean, small integer code -> ``nominal``. An order cannot
  be inferred statically (Q2 needs the user), so categoricals are never ``ordinal``;
- continuous numeric -> ``continuous``; timestamps -> ``continuous`` with ``datetime: True``;
- discrete numeric -> ``discrete`` with ``levels`` (enumerated domain) or ``step`` (lattice), or
  ``log_step`` for a lattice regular in orders of magnitude.

Ported from Hefesto's ``hefesto/core/typing/typing_inference.py``; decisions are identical.
Values are handled as strings with ``""`` as empty, as Hefesto reads them from CSV.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from functools import cached_property
from typing import TYPE_CHECKING, Any

from metis.infrastructure.typing.classification import (
    CONTINUOUS,
    MODE_ENUMERATED,
    MODE_LOG,
    Verdict,
    classify,
)
from metis.infrastructure.typing.domain import Domain, is_numeric, measure
from metis.infrastructure.typing.thresholds import InferenceThresholds

if TYPE_CHECKING:
    import pandas as pd

# A single format must explain ALL non-empty values of the column: avoids false positives on
# numeric columns ("20240101" matches none of these, having no separator).
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y")
# How datetime64 columns are rendered before inference (matches one of _DATE_FORMATS).
_DATETIME_RENDER = "%Y-%m-%d %H:%M:%S"

# File extensions that reveal a non-tabular blob: the column holds a file path or name.
_BLOB_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".svg",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".mp3",
    ".wav",
    ".mp4",
    ".avi",
    ".mov",
    ".exe",
    ".dll",
    ".bin",
)
_RE_BASE64 = re.compile(r"^[A-Za-z0-9+/]+=*$")
# Control characters except tab, newline and carriage return.
_RE_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MIN_BASE64_LENGTH = 500
_MAJORITY_RATIO = 0.5

# With exactly two values, quantity-vs-code and ordered-vs-unordered are the same trivial
# question: projected onto nominal without asking.
_BOOLEAN_VALUES = 2
# A nominal whose levels are all integers, with at least this many, may be a quantity.
_MIN_LEVELS_INTEGER_CODE = 3

NOTE_ID_CANDIDATE = (
    "doubt (id_candidate): one distinct value per row; could be an identifier. Typed 'id' "
    "(excluded from evaluation); declare it otherwise if it is a variable"
)
NOTE_UNDECIDED_BAND = (
    "doubt (undecided_band): regularity falls in the band where it does not decide; the "
    "conservative choice was applied"
)
NOTE_INTEGER_CODE = (
    "doubt (integer_code): nominal whose levels are all integers; it could be a quantity"
)
NOTE_NO_ORDER = (
    "an order cannot be inferred statically: typed nominal (declare 'ordinal' with its levels "
    "if the categories are ordered)"
)


@dataclass
class _Profile:
    """All this module needs to know about a column, counted ONCE.

    Every rule depends only on the distinct values and how many times each appears: a mean over
    rows is the ``counts``-weighted mean over distinct values, exactly. ``empty`` is counted
    apart because one rule looks at empties: an identifier requires one distinct value PER ROW.
    """

    n_total: int
    empty: int
    counts: Counter[str]

    @classmethod
    def of(cls, values: list[str]) -> _Profile:
        counts = Counter(values)
        empty = counts.pop("", 0)
        return cls(n_total=len(values), empty=empty, counts=counts)

    @property
    def n_present(self) -> int:
        return self.n_total - self.empty

    @cached_property
    def distinct(self) -> list[str]:
        """The non-empty distinct values, sorted."""
        return sorted(self.counts)

    def rows_where(self, predicate: Any) -> int:
        """How many ROWS satisfy ``predicate``, evaluating it once per distinct value."""
        return sum(c for v, c in self.counts.items() if predicate(v))

    def total(self, measure_of: Any) -> int:
        """Sum of ``measure_of`` over all rows, evaluated once per distinct value."""
        return sum(c * measure_of(v) for v, c in self.counts.items())

    @cached_property
    def domain(self) -> Domain | None:
        """The numeric domain measured on the rows, or ``None`` if the column is not numeric."""
        if not self.counts or not is_numeric(self.distinct):
            return None
        return measure(self.distinct, frequencies=self.counts)


def _is_empty(profile: _Profile) -> bool:
    return profile.n_present == 0


def _is_numeric_with_decimals(values: list[str]) -> bool:
    if not is_numeric(values):
        return False
    return any(Decimal(v) != Decimal(v).to_integral_value() for v in values)


def _is_identifier_candidate(profile: _Profile) -> bool:
    """One distinct value per row: may be an identifier, but the data cannot say so.

    Except if numeric with a fractional part: a measurement recorded at full precision repeats no
    value, and calling it an identifier left e.g. an unrounded Gaussian as a nominal with as many
    levels as rows. Integers without repeats remain candidates (``airbnb:id``,
    ``EmployeeNumber``); an integer quantity that never repeats is a declared limit.
    """
    distinct_with_empty = len(profile.counts) + (1 if profile.empty else 0)
    if profile.n_total <= 1 or distinct_with_empty != profile.n_total:
        return False
    return not _is_numeric_with_decimals(profile.distinct)


def _is_timestamp(profile: _Profile) -> bool:
    """Every non-empty value parses under the SAME fixed date format."""
    if not profile.counts:
        return False
    for fmt in _DATE_FORMATS:
        try:
            for value in profile.counts:
                datetime.strptime(value, fmt)  # noqa: DTZ007 (only checks parseability)
            return True
        except ValueError:
            continue
    return False


def _is_degenerate(profile: _Profile, threshold: int) -> bool:
    return len(profile.counts) <= threshold


def _has_blob_extension(value: str) -> bool:
    return value.lower().endswith(_BLOB_EXTENSIONS)


def _looks_like_long_base64(value: str) -> bool:
    return len(value) > _MIN_BASE64_LENGTH and bool(_RE_BASE64.fullmatch(value))


def _has_non_printable(value: str) -> bool:
    return _RE_NON_PRINTABLE.search(value) is not None


def _non_tabular_blob_reason(profile: _Profile) -> str | None:
    """The column stores a binary file path/name, a base64 blob, or control characters."""
    n = profile.n_present
    if not n:
        return None
    if profile.rows_where(_has_blob_extension) / n > _MAJORITY_RATIO:
        return "non-tabular blob: binary file extension in most values"
    if profile.rows_where(_looks_like_long_base64) / n > _MAJORITY_RATIO:
        return "non-tabular blob: long base64-compatible strings in most values"
    if profile.rows_where(_has_non_printable) / n > _MAJORITY_RATIO:
        return "non-tabular blob: non-printable content in most values"
    return None


def _free_text_reason(profile: _Profile, thresholds: InferenceThresholds) -> str | None:
    """Free text: a paragraph, not a code nor a label. Combined heuristic so that long
    categorical labels are not mistaken for it: a high mean word count, or a high mean length
    *together with* high uniqueness (free text rarely repeats; a long fixed label does)."""
    n = profile.n_present
    if not n:
        return None
    mean_words = profile.total(lambda v: len(v.split())) / n
    mean_chars = profile.total(len) / n
    uniqueness = len(profile.counts) / n
    is_free_text = mean_words > thresholds.free_text_words or (
        mean_chars > thresholds.free_text_length
        and uniqueness > thresholds.free_text_uniqueness_ratio
    )
    if not is_free_text:
        return None
    return (
        f"free text: mean {mean_words:.1f} words/value, {mean_chars:.0f} chars/value, "
        f"uniqueness ratio {uniqueness:.2f}"
    )


def _all_integers(values: list[str] | tuple[str, ...]) -> bool:
    """Only used to suspect a code: a category code never has a fractional part. It does NOT
    decide continuous vs discrete (there are integer continuous and fractional discrete)."""
    return all(Decimal(v) == Decimal(v).to_integral_value() for v in values)


def _nominal(levels: list[str], reason: str) -> tuple[dict[str, Any], list[str]]:
    notes = [reason]
    if len(levels) >= _MIN_LEVELS_INTEGER_CODE and is_numeric(levels) and _all_integers(levels):
        notes.append(NOTE_INTEGER_CODE)
    return {"type": "nominal"}, notes


def schema_entry(verdict: Verdict, domain: Domain) -> dict[str, Any]:
    """The METIS schema entry for a numeric verdict."""
    if verdict.type == CONTINUOUS:
        return {"type": "continuous"}
    mode = verdict.parameters.get("mode")
    if mode == MODE_ENUMERATED:
        levels = sorted(float(Decimal(v)) for v in domain.values)
        return {"type": "discrete", "step": None, "levels": levels}
    if mode == MODE_LOG:
        return {
            "type": "discrete",
            "step": None,
            "levels": None,
            "log_step": verdict.parameters.get("log_step"),
        }
    return {"type": "discrete", "step": verdict.parameters["step"], "levels": None}


def _classify_static(
    profile: _Profile, thresholds: InferenceThresholds
) -> tuple[dict[str, Any], list[str]]:
    """Everything that can be known without asking. Never leaves a column untyped."""
    levels = profile.distinct

    if not is_numeric(levels):
        return _nominal(levels, f"{len(levels)} observed levels; {NOTE_NO_ORDER}")

    domain = profile.domain
    assert domain is not None  # is_numeric(levels) just said so
    if _all_integers(domain.values) and domain.n_distinct <= thresholds.code_cardinality:
        return _nominal(
            levels,
            f"integer with {domain.n_distinct} distinct values (<= code_cardinality="
            f"{thresholds.code_cardinality}): treated as a category code (conservative; only "
            "the user can say it is a quantity)",
        )

    verdict = classify(domain, thresholds)
    entry = schema_entry(verdict, domain)
    notes = [verdict.evidence]
    if verdict.parameters.get("mode") == MODE_LOG:
        notes.append(
            "log lattice: 'log_step' is the common ratio between contiguous values "
            "(exp of the median log-gap); values lie on min * log_step**i"
        )
    if verdict.undecided:
        notes.append(NOTE_UNDECIDED_BAND)
    return entry, notes


def infer_column(
    values: list[str], thresholds: InferenceThresholds | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Type one column given as strings (``""`` = empty). Returns (schema entry, notes).

    Checks go from the most specific and deterministic (empty, blob, free text) to the most
    ambiguous (identifier, constant). Identifier is checked before timestamp on purpose: a column
    of unique timestamps is first an identifier doubt.
    """
    thresholds = thresholds or InferenceThresholds()
    profile = _Profile.of(values)

    if _is_empty(profile):
        return {"type": "id"}, [
            "empty column: every value is null or empty; excluded from evaluation"
        ]

    blob = _non_tabular_blob_reason(profile)
    if blob is not None:
        return {"type": "id"}, [f"{blob}; excluded from evaluation"]

    free_text = _free_text_reason(profile, thresholds)
    if free_text is not None:
        return {"type": "nominal"}, [free_text]

    if _is_identifier_candidate(profile):
        return {"type": "id"}, [
            f"{len(profile.distinct)} values, one distinct per row",
            NOTE_ID_CANDIDATE,
        ]

    if _is_timestamp(profile):
        return {"type": "continuous", "datetime": True}, [
            "timestamp: every value parses with the same date format"
        ]

    if _is_degenerate(profile, thresholds.degenerate):
        return {"type": "nominal"}, [
            f"constant: {len(profile.counts)} distinct value(s) in the sample (<= "
            f"degenerate={thresholds.degenerate})"
        ]

    if len(profile.counts) == _BOOLEAN_VALUES:
        return _nominal(profile.distinct, "boolean: two observed values, projected onto nominal")

    return _classify_static(profile, thresholds)


def _cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    if isinstance(value, str):
        return value
    try:
        import pandas as pd  # noqa: PLC0415

        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def column_as_strings(series: pd.Series) -> list[str]:
    """A DataFrame column as the strings the inference expects (``""`` for NaN/None/NaT).

    Floats use ``repr`` (shortest round-trip form); datetime64 columns are rendered as
    ``YYYY-MM-DD HH:MM:SS`` so the timestamp rule recognises them.
    """
    import pandas as pd  # noqa: PLC0415

    if pd.api.types.is_datetime64_any_dtype(series):
        rendered = series.dt.strftime(_DATETIME_RENDER)
        return ["" if pd.isna(v) else str(v) for v in rendered.tolist()]
    return [_cell_to_str(v) for v in series.tolist()]


@dataclass
class InferredSchema:
    """The inferred METIS schema plus, per column, why it was typed so and what remains in doubt."""

    schema: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)


def infer_schema(df: pd.DataFrame, thresholds: InferenceThresholds | None = None) -> InferredSchema:
    """Infer a METIS schema for every column of ``df`` (keys are ``str(column)``)."""
    thresholds = thresholds or InferenceThresholds()
    result = InferredSchema()
    for column in df.columns:
        name = str(column)
        entry, notes = infer_column(column_as_strings(df[column]), thresholds)
        result.schema[name] = entry
        result.notes[name] = notes
    return result
