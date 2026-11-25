"""Static column-type inference ported from Hefesto's Phase -1 (``metis.infrastructure.typing``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.typing import InferenceThresholds, infer_column, infer_schema
from metis.infrastructure.typing.inference import (
    NOTE_ID_CANDIDATE,
    NOTE_INTEGER_CODE,
    column_as_strings,
)

N = 500


@pytest.fixture(scope="module")
def inferred():
    rng = np.random.default_rng(0)
    words = [f"word{i}" for i in range(200)]
    df = pd.DataFrame(
        {
            "id": [f"user-{i:05d}" for i in range(N)],
            "const": ["same"] * N,
            "text": [" ".join(rng.choice(words, size=30)) for _ in range(N)],
            "colour": rng.choice(["red", "green", "blue", "black"], size=N),
            "gauss": rng.normal(170.0, 10.0, size=N),
            "count": rng.integers(0, 15, size=N),
            "half": rng.integers(0, 200, size=N) * 0.5,
            "date": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 60, size=N), unit="D"),
        }
    )
    return infer_schema(df)


def test_identifier_candidate_is_id(inferred):
    assert inferred.schema["id"] == {"type": "id"}
    assert NOTE_ID_CANDIDATE in inferred.notes["id"]


def test_constant_is_nominal(inferred):
    assert inferred.schema["const"] == {"type": "nominal"}
    assert inferred.notes["const"][0].startswith("constant")


def test_free_text_is_nominal(inferred):
    assert inferred.schema["text"] == {"type": "nominal"}
    assert inferred.notes["text"][0].startswith("free text")


def test_categorical_is_nominal_without_order(inferred):
    assert inferred.schema["colour"] == {"type": "nominal"}
    assert "order cannot be inferred" in inferred.notes["colour"][0]


def test_full_precision_gaussian_is_continuous(inferred):
    # Unique values with decimals are a measurement, not an identifier candidate.
    assert inferred.schema["gauss"] == {"type": "continuous"}


def test_integer_count_is_enumerated(inferred):
    entry = inferred.schema["count"]
    assert entry["type"] == "discrete"
    assert entry["step"] is None
    assert entry["levels"] == [float(v) for v in range(15)]


def test_half_step_lattice(inferred):
    assert inferred.schema["half"] == {"type": "discrete", "step": 0.5, "levels": None}


def test_dates_are_continuous_datetime(inferred):
    assert inferred.schema["date"] == {"type": "continuous", "datetime": True}


def test_small_integer_code_is_nominal_with_doubt():
    entry, notes = infer_column([str(v % 5 + 1) for v in range(100)])
    assert entry == {"type": "nominal"}
    assert NOTE_INTEGER_CODE in notes


def test_boolean_is_nominal():
    entry, _ = infer_column(["yes", "no"] * 10)
    assert entry == {"type": "nominal"}


def test_empty_column_is_excluded():
    entry, notes = infer_column(["", ""])
    assert entry == {"type": "id"}
    assert notes[0].startswith("empty column")


def test_log_lattice_gives_log_step():
    values = [str(10**k) for k in range(25)] * 4
    entry, _ = infer_column(values)
    assert entry["type"] == "discrete"
    assert entry["log_step"] == pytest.approx(10.0)


def test_nan_and_none_are_empty():
    series = pd.Series([1.5, np.nan, None, 2.0], dtype=object)
    assert column_as_strings(series) == ["1.5", "", "", "2.0"]


def test_thresholds_accept_hefesto_keys_and_validate_band():
    thresholds = InferenceThresholds.from_mapping({"umbral_enumerable": 5, "lattice": 0.9})
    assert thresholds.enumerable == 5
    assert thresholds.lattice == 0.9
    with pytest.raises(ValueError, match="no_lattice"):
        InferenceThresholds(lattice=0.3, no_lattice=0.4)
