"""Tests for metis.shared.multi_run_stats — cross-run statistical aggregation.

WHY: Multi-run mode computes median/mean/std across seeds for stability
analysis. Wrong statistics = wrong confidence in evaluation results.
"""

import pytest

from metis.shared.multi_run_stats import _calculate_score_stats, calculate_multi_run_statistics


class TestCalculateMultiRunStatistics:
    """Risk: incorrect aggregation across runs invalidates reproducibility claims."""

    def test_basic_aggregation(self):
        runs = [
            {"scores": {"fidelity_score": 0.70, "composite_score": 0.65}},
            {"scores": {"fidelity_score": 0.80, "composite_score": 0.75}},
            {"scores": {"fidelity_score": 0.75, "composite_score": 0.70}},
        ]
        stats = calculate_multi_run_statistics(runs)

        assert "fidelity_score" in stats
        assert stats["fidelity_score"]["median"] == pytest.approx(0.75)
        assert stats["fidelity_score"]["mean"] == pytest.approx(0.75)
        assert stats["fidelity_score"]["min"] == pytest.approx(0.70)
        assert stats["fidelity_score"]["max"] == pytest.approx(0.80)
        assert stats["fidelity_score"]["n_runs"] == 3

    def test_single_run_std_zero(self):
        runs = [{"scores": {"fidelity_score": 0.80}}]
        stats = calculate_multi_run_statistics(runs)
        assert stats["fidelity_score"]["std"] == 0.0
        assert stats["fidelity_score"]["n_runs"] == 1

    def test_empty_input(self):
        assert calculate_multi_run_statistics([]) == {}

    def test_non_score_keys_ignored(self):
        """Only keys ending in '_score' are aggregated."""
        runs = [
            {"scores": {"fidelity_score": 0.8, "total_metrics": 10}},
            {"scores": {"fidelity_score": 0.7, "total_metrics": 10}},
        ]
        stats = calculate_multi_run_statistics(runs)
        assert "fidelity_score" in stats
        assert "total_metrics" not in stats

    def test_quantiles(self):
        runs = [{"scores": {"composite_score": v}} for v in [0.1, 0.3, 0.5, 0.7, 0.9]]
        stats = calculate_multi_run_statistics(runs)
        assert stats["composite_score"]["q1"] == pytest.approx(0.3)
        assert stats["composite_score"]["q3"] == pytest.approx(0.7)

    def test_missing_score_in_some_runs(self):
        """If a score is absent in some runs, only available values are used."""
        runs = [
            {"scores": {"fidelity_score": 0.8}},
            {"scores": {"fidelity_score": 0.6}},
            {"scores": {}},
        ]
        stats = calculate_multi_run_statistics(runs)
        assert stats["fidelity_score"]["n_runs"] == 2


class TestCalculateScoreStats:
    """Internal helper for per-score statistics."""

    def test_basic(self):
        stats = _calculate_score_stats([0.5, 0.6, 0.7])
        assert stats["mean"] == pytest.approx(0.6)
        assert stats["median"] == pytest.approx(0.6)
        assert stats["min"] == pytest.approx(0.5)
        assert stats["max"] == pytest.approx(0.7)

    def test_single_value_std_zero(self):
        stats = _calculate_score_stats([0.42])
        assert stats["std"] == 0.0
        assert stats["mean"] == pytest.approx(0.42)
