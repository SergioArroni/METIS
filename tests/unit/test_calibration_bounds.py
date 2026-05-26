"""Tests for metis.calibrate.core.bounds — CalibrationBounds storage.

WHY: CalibrationBounds is the persistence layer for calibration results.
Normalization math, save/load JSON roundtrip, and inversion handling
are all critical. A broken bounds file = wrong normalized scores everywhere.
"""

import json
import warnings

import pytest

from metis.calibrate.core.bounds import CalibrationBounds

# =============================================================================
# Basic set/get operations
# =============================================================================


class TestBoundsSetGet:
    def test_set_and_get(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", lower_bound=0.3, upper_bound=0.9)
        lower, upper = bounds.get_bounds("fidelity")
        assert lower == 0.3
        assert upper == 0.9

    def test_get_nonexistent_raises(self):
        bounds = CalibrationBounds()
        with pytest.raises(KeyError, match="fidelity"):
            bounds.get_bounds("fidelity")

    def test_get_all_families(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", 0.2, 0.8)
        bounds.set_bounds("privacy", 0.3, 0.7)
        families = bounds.get_all_families()
        assert set(families) == {"fidelity", "privacy"}

    def test_with_iterations(self):
        bounds = CalibrationBounds()
        bounds.set_bounds(
            "fidelity",
            lower_bound=0.3,
            upper_bound=0.9,
            lower_iterations=[0.28, 0.30, 0.32],
            upper_iterations=[0.88, 0.90, 0.92],
        )
        assert bounds.bounds["fidelity"]["lower_iterations"] == [0.28, 0.30, 0.32]

    def test_inverted_flag(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("privacy", 0.2, 0.8, inverted=True)
        assert bounds.bounds["privacy"]["inverted"] is True

    def test_invalid_bounds_warns(self):
        """Upper < lower should emit a warning (caller error)."""
        bounds = CalibrationBounds()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bounds.set_bounds("fidelity", lower_bound=0.9, upper_bound=0.3)
            assert len(w) == 1
            assert "INVALID BOUNDS" in str(w[0].message)


# =============================================================================
# normalize_with_bounds — the core math
# =============================================================================


class TestNormalizeWithBounds:
    """Risk: wrong normalization formula = all calibrated scores are meaningless."""

    def test_at_lower_bound(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", lower_bound=0.3, upper_bound=0.9)
        result = bounds.normalize_with_bounds("fidelity", 0.3)
        assert result == pytest.approx(0.0)

    def test_at_upper_bound(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", lower_bound=0.3, upper_bound=0.9)
        result = bounds.normalize_with_bounds("fidelity", 0.9)
        assert result == pytest.approx(1.0)

    def test_midpoint(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", lower_bound=0.0, upper_bound=1.0)
        result = bounds.normalize_with_bounds("fidelity", 0.5)
        assert result == pytest.approx(0.5)

    def test_below_lower_clips_to_zero(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", lower_bound=0.3, upper_bound=0.9)
        result = bounds.normalize_with_bounds("fidelity", 0.1)
        assert result == 0.0

    def test_above_upper_clips_to_one(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", lower_bound=0.3, upper_bound=0.9)
        result = bounds.normalize_with_bounds("fidelity", 0.95)
        assert result == 1.0

    def test_identical_bounds_edge_case(self):
        """When lower == upper, should return 1 if >= upper, else 0."""
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", lower_bound=0.5, upper_bound=0.5)
        assert bounds.normalize_with_bounds("fidelity", 0.5) == 1.0
        assert bounds.normalize_with_bounds("fidelity", 0.4) == 0.0

    def test_inverted_flips_result(self):
        """When inverted=True, 1-normalized is returned."""
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", lower_bound=0.2, upper_bound=0.8, inverted=True)
        # Raw 0.8 → normalized (0.8-0.2)/(0.8-0.2) = 1.0 → inverted = 0.0
        result = bounds.normalize_with_bounds("fidelity", 0.8)
        assert result == pytest.approx(0.0)
        # Raw 0.2 → normalized 0.0 → inverted = 1.0
        result = bounds.normalize_with_bounds("fidelity", 0.2)
        assert result == pytest.approx(1.0)

    def test_unknown_family_raises(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", 0.2, 0.8)
        with pytest.raises(ValueError, match="No calibration bounds"):
            bounds.normalize_with_bounds("utility", 0.5)


# =============================================================================
# Save / Load JSON roundtrip
# =============================================================================


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        filepath = str(tmp_path / "bounds.json")

        original = CalibrationBounds()
        original.set_bounds("fidelity", 0.25, 0.85, lower_iterations=[0.24, 0.26])
        original.set_bounds("privacy", 0.30, 0.70, inverted=True)
        original.set_metadata("dataset_rows", 1000)
        original.optimal_aggregators = {"composite": "ssd"}
        original.save(filepath)

        loaded = CalibrationBounds.load(filepath)
        assert loaded.get_bounds("fidelity") == (0.25, 0.85)
        assert loaded.get_bounds("privacy") == (0.30, 0.70)
        assert loaded.bounds["privacy"]["inverted"] is True
        assert loaded.get_metadata("dataset_rows") == 1000
        assert loaded.optimal_aggregators == {"composite": "ssd"}

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="Calibration file not found"):
            CalibrationBounds.load("/nonexistent/path.json")

    def test_load_invalid_format(self, tmp_path):
        filepath = tmp_path / "bad.json"
        filepath.write_text("[]")  # Array instead of object
        with pytest.raises(ValueError, match="JSON object"):
            CalibrationBounds.load(str(filepath))

    def test_load_empty_bounds(self, tmp_path):
        filepath = tmp_path / "empty.json"
        filepath.write_text(json.dumps({"bounds": {}, "metadata": {}}))
        with pytest.raises(ValueError, match="no bounds data"):
            CalibrationBounds.load(str(filepath))

    def test_load_invalid_bounds_structure(self, tmp_path):
        filepath = tmp_path / "bad_bounds.json"
        filepath.write_text(json.dumps({"bounds": "not_a_dict", "metadata": {}}))
        with pytest.raises(ValueError, match="must be a JSON object"):
            CalibrationBounds.load(str(filepath))


# =============================================================================
# Metadata and utilities
# =============================================================================


class TestMetadata:
    def test_set_and_get_metadata(self):
        bounds = CalibrationBounds()
        bounds.set_metadata("version", "1.0")
        assert bounds.get_metadata("version") == "1.0"

    def test_get_metadata_default(self):
        bounds = CalibrationBounds()
        assert bounds.get_metadata("missing", default="fallback") == "fallback"

    def test_len(self):
        bounds = CalibrationBounds()
        assert len(bounds) == 0
        bounds.set_bounds("fidelity", 0.2, 0.8)
        assert len(bounds) == 1

    def test_contains(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", 0.2, 0.8)
        assert "fidelity" in bounds
        assert "privacy" not in bounds

    def test_repr(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", 0.2, 0.8)
        assert "fidelity" in repr(bounds)

    def test_to_dict(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", 0.2, 0.8)
        bounds.set_metadata("seed", 42)
        d = bounds.to_dict()
        assert "bounds" in d
        assert "metadata" in d
        assert d["bounds"]["fidelity"]["lower"] == 0.2

    def test_get_summary_string(self):
        bounds = CalibrationBounds()
        bounds.set_bounds("fidelity", 0.25, 0.85)
        summary = bounds.get_summary()
        assert "FIDELITY" in summary
        assert "0.25" in summary
        assert "0.85" in summary
