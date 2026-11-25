from pathlib import Path

import numpy as np
import pandas as pd

from metis.sota_models.comparison import compare_across_datasets
from metis.sota_models.comparison.reporter import BenchmarkReporter
from metis.sota_models.comparison.statistical_tests import FriedmanNemenyiTest


def _scores(**gens):
    return {
        g: {
            seed: {"aggregated_scores": dict(zip(("fidelity", "utility", "privacy", "overall"), v))}
            for seed, v in seeds.items()
        }
        for g, seeds in gens.items()
    }


def test_tables_use_the_measured_scores_not_display_anchors(tmp_path: Path) -> None:
    """real_data/uniform_noise used to be forced to 1/0 in the tables only (B7)."""
    results = _scores(
        real_data={42: (0.97, 0.99, 0.98, 0.98), 43: (0.96, 1.0, 0.99, 0.98)},
        uniform_noise={42: (0.05, 0.0, 1.0, 0.3), 43: (0.04, 0.0, 1.0, 0.3)},
    )
    reporter = BenchmarkReporter(results=results, output_dir=str(tmp_path / "benchmark_airbnb"))
    summary = reporter._calculate_summary_stats(reporter._extract_scores()).set_index("generator")
    assert summary.loc["real_data", "privacy_mean"] == np.mean([0.98, 0.99])
    assert summary.loc["uniform_noise", "fidelity_mean"] == np.mean([0.05, 0.04])
    assert not hasattr(reporter, "_adjust_display_scores")


def test_failed_seeds_are_counted_not_silently_dropped(tmp_path: Path) -> None:
    results = _scores(a={1: (0.5, 0.5, 0.5, 0.5), 2: (0.6, 0.6, 0.6, 0.6)})
    results["a"][3] = {"error": "boom"}
    reporter = BenchmarkReporter(results=results, output_dir=str(tmp_path))
    summary = reporter._calculate_summary_stats(reporter._extract_scores()).set_index("generator")
    assert summary.loc["a", "n_seeds"] == 2
    assert summary.loc["a", "failed_seeds"] == [3]


def test_friedman_reports_kendall_w_and_leaves_out_incomplete_blocks() -> None:
    scores = pd.DataFrame(
        {"g1": [0.9, 0.8, 0.85, np.nan], "g2": [0.5, 0.4, 0.45, 0.5], "g3": [0.1, 0.2, 0.15, 0.1]},
        index=[0, 1, 2, 3],
    )
    res = FriedmanNemenyiTest(block_label="seed").compare(scores)
    assert res.n_blocks == 3
    assert res.dropped_blocks == [3]
    # Identical order in every block: perfect concordance.
    assert res.kendall_w == 1.0
    assert "Left out" in res.interpretation


def test_cross_dataset_friedman_uses_datasets_as_blocks(tmp_path: Path) -> None:
    """Seeds as blocks only say the ranking is stable on one dataset; the
    generalisation test blocks on datasets (Demšar 2006)."""
    rng = np.random.default_rng(0)
    by_dataset = {}
    for d in range(5):
        gens = {}
        for g, level in (("a", 0.9), ("b", 0.6), ("c", 0.3)):
            gens[g] = {s: tuple(level + 0.01 * rng.standard_normal(4)) for s in range(3)}
        by_dataset[f"ds{d}"] = _scores(**gens)
    res = compare_across_datasets(by_dataset, str(tmp_path))
    assert res["overall"].n_blocks == 5
    assert "datasets" in res["overall"].interpretation
    assert (tmp_path / "cross_dataset_comparison.json").exists()
    md = (tmp_path / "cross_dataset_comparison.md").read_text(encoding="utf-8")
    assert "datasets as blocks" in md and "| a |" in md


def test_seed_blocks_say_they_measure_stability(tmp_path: Path) -> None:
    pivot = pd.DataFrame(
        {"a": [0.9, 0.91, 0.92], "b": [0.5, 0.52, 0.51], "c": [0.1, 0.12, 0.11]}, index=[42, 43, 44]
    )
    res = FriedmanNemenyiTest(block_label="seed").compare(pivot)
    assert "stable across runs" in res.interpretation
    res = FriedmanNemenyiTest(block_label="dataset").compare(pivot)
    assert "stable across runs" not in res.interpretation
