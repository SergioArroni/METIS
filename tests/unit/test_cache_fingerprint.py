"""Regression tests for calibration cache fingerprinting.

These tests defend the security invariants documented in
``/memories/repo/paper_review_findings.md`` and the Phase 4 plan:

* The data fingerprint MUST cover the full DataFrame, not a 1000-row
  sample. Two datasets that differ only in non-sampled rows must produce
  different fingerprints (cache-poisoning protection).
* Hash components must remain at 32 hex chars (128-bit) — long enough to
  resist collision attacks, short enough to keep cache filenames inside
  Windows MAX_PATH.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metis.calibrate.cache.fingerprint import compute_data_fingerprint, generate_cache_key


def _make_large_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.integers(0, 5, size=n),
            "c": rng.choice(["x", "y", "z"], size=n),
        }
    )


def test_data_fingerprint_detects_changes_outside_first_1000_rows():
    """A change in row 5000 must alter the fingerprint (full hash, not sampled)."""
    df1 = _make_large_df(6000, seed=1)
    df2 = df1.copy()
    df2.loc[5000, "a"] = df1.loc[5000, "a"] + 1e6  # tampered row beyond 1000-row sample

    fp1 = compute_data_fingerprint(df1)
    fp2 = compute_data_fingerprint(df2)

    assert fp1 != fp2, "fingerprint must detect tampering beyond row 1000 (cache poisoning)"


def test_data_fingerprint_components_are_32_hex():
    df = _make_large_df(50)
    fp = compute_data_fingerprint(df)
    parts = fp.split("_")
    assert len(parts) == 3, f"expected 3 hash components, got {parts}"
    for component in parts:
        assert len(component) == 32, f"expected 128-bit prefix, got {len(component)} chars"
        int(component, 16)  # must be valid hex


def test_data_fingerprint_deterministic():
    df = _make_large_df(2000, seed=42)
    assert compute_data_fingerprint(df) == compute_data_fingerprint(df.copy())


def test_generate_cache_key_changes_with_params():
    base = generate_cache_key("d" * 32, "c" * 32, n_iterations=12, sample_size=500, base_seed=42)
    other = generate_cache_key("d" * 32, "c" * 32, n_iterations=13, sample_size=500, base_seed=42)
    assert base != other
