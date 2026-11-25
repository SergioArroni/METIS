"""Theil's U is asymmetric: both directions are measured (2026-10-19).

Before, only U(second | first) was measured, with the columns in alphabetical
order, so whether a one-direction failure was seen depended on the names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metis.infrastructure.metrics.fidelity.conditional.cat_cat.theils_u import TheilsUMetric

CITIES = {"FR": ["Paris", "Lyon", "Lille"], "ES": ["Madrid", "Sevilla", "Bilbao"]}
WEIGHTS = [0.8, 0.15, 0.05]  # within a country the first city dominates


def _tables(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    country = rng.choice(list(CITIES), n)
    real_city = [rng.choice(CITIES[c], p=WEIGHTS) for c in country]
    # Keeps city -> country exactly, but the cities within a country come out uniform.
    synth_city = [rng.choice(CITIES[c]) for c in country]
    return (
        pd.DataFrame({"city": real_city, "country": country}),
        pd.DataFrame({"city": synth_city, "country": country}),
    )


def _score(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    return TheilsUMetric().fit(real, synth, {}).compute().value


def test_both_directions_are_measured():
    real, synth = _tables()
    details = TheilsUMetric().fit(real, synth, {}).compute().details
    assert {"city|country", "country|city"} <= set(details)


def test_a_one_direction_failure_is_seen_whatever_the_column_names():
    real, synth = _tables()
    score = _score(real, synth)
    # Renaming reverses the alphabetical order of the two columns.
    renamed = {"city": "zz_city", "country": "aa_country"}
    assert _score(real.rename(columns=renamed), synth.rename(columns=renamed)) == pytest.approx(
        score
    )
    assert score < 0.95
    same = _tables(seed=0)[0]
    assert _score(real, same) == pytest.approx(1.0)
