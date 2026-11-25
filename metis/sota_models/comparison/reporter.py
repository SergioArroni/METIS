"""
Reporter for benchmark comparison results.

This module generates comprehensive comparison reports including:
- Ranking tables across dimensions (fidelity, utility, privacy)
- Statistical significance analysis (Friedman + Nemenyi)
- Summary statistics (mean, std, median, IQR)
- Critical Difference diagrams
- Markdown and JSON outputs
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .statistical_tests import TestResult, get_statistical_test

# Tables, rankings and tests are all computed from the same measured scores.
# There used to be "display anchors" (real_data forced to fidelity = utility = 1
# and privacy = 0, uniform_noise to the opposite) applied to the tables only,
# and a different composite formula for Airbnb only: tables and rankings did
# not describe the same numbers.


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BenchmarkReporter:
    """
    Reporter for generating benchmark comparison reports.

    Takes results from BenchmarkOrchestrator and produces:
    - Ranking tables
    - Statistical significance tests
    - Summary markdown report
    - Detailed JSON results
    """

    def __init__(self, results: dict, output_dir: str):
        """
        Initialize benchmark reporter.

        Args:
            results: Results dictionary from BenchmarkOrchestrator
            output_dir: Directory to save reports
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(self, statistical_test: str = "friedman-nemenyi", alpha: float = 0.05):
        """
        Generate all comparison reports.

        Args:
            statistical_test: Name of statistical test to use
            alpha: Significance level
        """
        print("\nGenerating benchmark comparison reports...")

        # Extract scores into structured format
        scores_df = self._extract_scores()

        if scores_df.empty:
            print("No valid results to report")
            return

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(scores_df)

        # Calculate rankings
        rankings = self._calculate_rankings(scores_df)

        # Perform statistical tests
        test_results = self._perform_statistical_tests(scores_df, statistical_test, alpha)

        # Generate markdown report
        self._generate_markdown_report(summary_stats, rankings, test_results)

        # Generate JSON report
        self._generate_json_report(summary_stats, rankings, test_results)

        # Generate CSV exports
        self._generate_csv_exports(scores_df, summary_stats, rankings)

        print(f"Reports saved to: {self.output_dir}")

    def _extract_scores(self) -> pd.DataFrame:
        """
        Extract scores from results into structured DataFrame.

        Returns:
            DataFrame with columns: generator, seed, fidelity, utility, privacy, overall
            plus optional raw (pre-calibration) columns when available.
        """
        rows = []

        self.failed_runs = {}
        for gen_name, gen_results in self.results.items():
            failed = [seed for seed, r in gen_results.items() if "error" in r]
            if failed:
                # Reported in the summary; they used to vanish from the means.
                self.failed_runs[gen_name] = failed
            for seed, seed_results in gen_results.items():
                if "error" not in seed_results:
                    # Support both formats: new format (aggregated_scores) and old (run_summary).
                    # A missing family is NaN (it used to be 0.0, which ranked
                    # the generator last on a family that was never measured).
                    if "aggregated_scores" in seed_results:
                        aggregates = seed_results["aggregated_scores"]
                        row = {
                            "generator": gen_name,
                            "seed": seed,
                            "fidelity": aggregates.get("fidelity", np.nan),
                            "utility": aggregates.get("utility", np.nan),
                            "privacy": aggregates.get("privacy", np.nan),
                            "overall": aggregates.get("overall", np.nan),
                        }
                        if aggregates.get("calibrated"):
                            row["fidelity_raw"] = aggregates.get("fidelity_raw")
                            row["utility_raw"] = aggregates.get("utility_raw")
                            row["privacy_raw"] = aggregates.get("privacy_raw")
                        rows.append(row)
                    elif "run_summary" in seed_results:
                        run_summary = seed_results["run_summary"]
                        # Handle both dict (from JSON) and RunSummary object
                        if hasattr(run_summary, "aggregates"):
                            aggregates = run_summary.aggregates
                        else:
                            aggregates = run_summary.get("aggregates", {})
                        row = {
                            "generator": gen_name,
                            "seed": seed,
                            "fidelity": aggregates.get("fidelity_score", np.nan),
                            "utility": aggregates.get("utility_score", np.nan),
                            "privacy": aggregates.get("privacy_score", np.nan),
                            "overall": aggregates.get("composite_score", np.nan),
                        }
                        if aggregates.get("composite_score_calibrated"):
                            row["fidelity_raw"] = aggregates.get("fidelity_score_raw")
                            row["utility_raw"] = aggregates.get("utility_score_raw")
                            row["privacy_raw"] = aggregates.get("privacy_score_raw")
                        rows.append(row)

        return pd.DataFrame(rows)

    def _calculate_summary_stats(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics for each generator.

        Args:
            scores_df: DataFrame with scores

        Returns:
            DataFrame with summary statistics
        """
        dimensions = ["fidelity", "utility", "privacy", "overall"]
        has_raw = "fidelity_raw" in scores_df.columns

        summary_rows = []

        for gen_name in scores_df["generator"].unique():
            gen_scores = scores_df[scores_df["generator"] == gen_name]

            row = {
                "generator": gen_name,
                "n_seeds": int(len(gen_scores)),
                "failed_seeds": list(getattr(self, "failed_runs", {}).get(gen_name, [])),
            }

            for dim in dimensions:
                values = gen_scores[dim]
                row[f"{dim}_mean"] = values.mean()
                row[f"{dim}_std"] = values.std()
                row[f"{dim}_median"] = values.median()
                row[f"{dim}_q25"] = values.quantile(0.25)
                row[f"{dim}_q75"] = values.quantile(0.75)
                row[f"{dim}_iqr"] = row[f"{dim}_q75"] - row[f"{dim}_q25"]

                # Include raw score stats when available
                raw_col = f"{dim}_raw"
                if has_raw and raw_col in gen_scores.columns:
                    raw_values = gen_scores[raw_col].dropna()
                    if not raw_values.empty:
                        row[f"{dim}_raw_mean"] = raw_values.mean()
                        row[f"{dim}_raw_std"] = raw_values.std()

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def _calculate_rankings(self, scores_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Calculate rankings for each dimension.

        Args:
            scores_df: DataFrame with scores

        Returns:
            Dictionary mapping dimension names to ranking DataFrames
        """
        dimensions = ["fidelity", "utility", "privacy", "overall"]
        rankings = {}

        for dim in dimensions:
            # Pivot to get generators as columns, seeds as rows
            pivot = scores_df.pivot(index="seed", columns="generator", values=dim)

            # Rank each row (higher score = better rank = rank 1)
            ranked = pivot.rank(axis=1, ascending=False, method="average")

            # Calculate average rank for each generator
            avg_ranks = ranked.mean().sort_values()

            rankings[dim] = {
                "scores": pivot,
                "ranks": ranked,
                "avg_ranks": avg_ranks,
            }

        return rankings

    def _perform_statistical_tests(
        self, scores_df: pd.DataFrame, test_name: str, alpha: float
    ) -> dict[str, TestResult]:
        """
        Perform statistical significance tests.

        Args:
            scores_df: DataFrame with scores
            test_name: Name of statistical test
            alpha: Significance level

        Returns:
            Dictionary mapping dimension names to TestResult objects
        """
        test = get_statistical_test(test_name, alpha, block_label="seed")
        dimensions = ["fidelity", "utility", "privacy", "overall"]

        test_results = {}

        for dim in dimensions:
            # Pivot scores for this dimension
            pivot = scores_df.pivot(index="seed", columns="generator", values=dim)

            # Run statistical test
            result = test.compare(pivot, metric_name=dim)
            test_results[dim] = result

        return test_results

    def _generate_markdown_report(
        self,
        summary_stats: pd.DataFrame,
        rankings: dict[str, pd.DataFrame],
        test_results: dict[str, TestResult],
    ):
        """Generate markdown report."""
        report_path = self.output_dir / "benchmark_comparison.md"

        with report_path.open("w", encoding="utf-8") as f:
            f.write("# Benchmark Comparison Report\n\n")

            # Summary section
            f.write("## Summary Statistics\n\n")
            f.write("Mean scores (± std) for each generator across all runs:\n\n")

            for _, row in summary_stats.iterrows():
                f.write(f"### {row['generator']}\n\n")
                f.write(f"- **Fidelity**: {row['fidelity_mean']:.4f} ± {row['fidelity_std']:.4f}")
                if "fidelity_raw_mean" in row and pd.notna(row.get("fidelity_raw_mean")):
                    f.write(f" (raw: {row['fidelity_raw_mean']:.4f})")
                f.write("\n")
                f.write(f"- **Utility**: {row['utility_mean']:.4f} ± {row['utility_std']:.4f}")
                if "utility_raw_mean" in row and pd.notna(row.get("utility_raw_mean")):
                    f.write(f" (raw: {row['utility_raw_mean']:.4f})")
                f.write("\n")
                f.write(f"- **Privacy**: {row['privacy_mean']:.4f} ± {row['privacy_std']:.4f}")
                if "privacy_raw_mean" in row and pd.notna(row.get("privacy_raw_mean")):
                    f.write(f" (raw: {row['privacy_raw_mean']:.4f})")
                f.write("\n")
                f.write(f"- **Overall**: {row['overall_mean']:.4f} ± {row['overall_std']:.4f}\n\n")

            # Rankings section
            f.write("## Rankings\n\n")
            f.write("Average ranks (1=best) for each dimension:\n\n")

            for dim in ["fidelity", "utility", "privacy", "overall"]:
                f.write(f"### {dim.capitalize()}\n\n")
                avg_ranks = rankings[dim]["avg_ranks"]

                f.write("| Rank | Generator | Avg Rank |\n")
                f.write("|------|-----------|----------|\n")

                for rank, (gen_name, avg_rank) in enumerate(avg_ranks.items(), 1):
                    f.write(f"| {rank} | {gen_name} | {avg_rank:.2f} |\n")

                f.write("\n")

            # Statistical tests section
            f.write("## Statistical Significance Tests (seeds as blocks)\n\n")
            f.write(
                "The blocks are the seeds of this dataset: these tests say whether the "
                "ranking is stable across runs, not whether it generalises to other "
                "datasets. For that use `compare_across_datasets` (datasets as blocks, "
                "Demšar 2006).\n\n"
            )

            for dim in ["fidelity", "utility", "privacy", "overall"]:
                result = test_results[dim]

                f.write(f"### {dim.capitalize()}\n\n")
                f.write(f"**Test**: {result.test_name}\n\n")
                f.write(f"**Statistic**: {result.statistic:.4f}\n\n")
                f.write(f"**p-value**: {result.p_value:.4f}\n\n")
                if result.kendall_w is not None:
                    f.write(f"**Kendall's W**: {result.kendall_w:.4f}\n\n")
                f.write(
                    f"**Significant**: {'Yes' if result.is_significant else 'No'} (α={result.alpha})\n\n"
                )
                f.write(f"**Interpretation**: {result.interpretation}\n\n")

                if result.pairwise_comparisons is not None:
                    f.write("#### Pairwise Comparisons\n\n")
                    f.write("Pairs with significant differences:\n\n")

                    significant = result.pairwise_comparisons[
                        result.pairwise_comparisons["significant"]
                    ]

                    if len(significant) > 0:
                        f.write("| Method 1 | Method 2 | Rank Diff | Significant |\n")
                        f.write("|----------|----------|-----------|-------------|\n")

                        for _, comp in significant.iterrows():
                            f.write(
                                f"| {comp['method_1']} | {comp['method_2']} | "
                                f"{comp['rank_diff']:.2f} | ✓ |\n"
                            )
                    else:
                        f.write("No significant pairwise differences found.\n")

                    f.write("\n")

            # Best generators section
            f.write("## Best Generators by Dimension\n\n")

            for dim in ["fidelity", "utility", "privacy", "overall"]:
                best_gen = rankings[dim]["avg_ranks"].idxmin()
                best_rank = rankings[dim]["avg_ranks"].min()
                f.write(f"- **{dim.capitalize()}**: {best_gen} (rank: {best_rank:.2f})\n")

            f.write("\n")

    def _generate_json_report(
        self,
        summary_stats: pd.DataFrame,
        rankings: dict[str, pd.DataFrame],
        test_results: dict[str, TestResult],
    ):
        """Generate JSON report."""
        report_path = self.output_dir / "benchmark_comparison.json"

        report = {
            "failed_runs": getattr(self, "failed_runs", {}),
            "summary_statistics": summary_stats.to_dict(orient="records"),
            "rankings": {},
            "statistical_tests": {},
        }

        # Add rankings
        for dim in ["fidelity", "utility", "privacy", "overall"]:
            report["rankings"][dim] = {
                "average_ranks": rankings[dim]["avg_ranks"].to_dict(),
            }

        # Add test results
        for dim in ["fidelity", "utility", "privacy", "overall"]:
            result = test_results[dim]
            report["statistical_tests"][dim] = {
                "test_name": result.test_name,
                "statistic": (float(result.statistic) if not np.isnan(result.statistic) else None),
                "p_value": (float(result.p_value) if not np.isnan(result.p_value) else None),
                "is_significant": bool(result.is_significant),
                "alpha": float(result.alpha),
                "critical_difference": (
                    float(result.critical_difference) if result.critical_difference else None
                ),
                "interpretation": result.interpretation,
                "kendall_w": result.kendall_w,
                "n_blocks": result.n_blocks,
                "dropped_blocks": result.dropped_blocks,
                "dropped_methods": result.dropped_methods,
            }

            if result.pairwise_comparisons is not None:
                report["statistical_tests"][dim]["pairwise_comparisons"] = (
                    result.pairwise_comparisons.to_dict(orient="records")
                )

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, cls=NumpyJSONEncoder)

    def _generate_csv_exports(
        self,
        scores_df: pd.DataFrame,
        summary_stats: pd.DataFrame,
        rankings: dict[str, pd.DataFrame],
    ):
        """Generate CSV exports."""
        # Raw scores
        scores_df.to_csv(self.output_dir / "scores_raw.csv", index=False)

        # Summary statistics
        summary_stats.to_csv(self.output_dir / "summary_statistics.csv", index=False)

        # Rankings
        for dim in ["fidelity", "utility", "privacy", "overall"]:
            avg_ranks = rankings[dim]["avg_ranks"]
            avg_ranks.to_csv(self.output_dir / f"rankings_{dim}.csv", header=["average_rank"])


def compare_across_datasets(
    results_by_dataset: dict[str, dict],
    output_dir: str,
    alpha: float = 0.05,
) -> dict[str, TestResult]:
    """Friedman + Nemenyi with the datasets as blocks (Demšar 2006).

    ``results_by_dataset`` maps each dataset to the results dictionary that
    ``BenchmarkReporter`` takes.  Each generator's score on a dataset is the
    mean over its seeds; the test then asks whether some generators are
    better than others *across datasets*, which the per-dataset reports
    (seeds as blocks) cannot.  With few datasets the critical difference is
    large (k = 12, N = 4: 8.3 ranks) and only the extremes separate; the
    interpretation carries the low-power warning of the test.

    Writes ``cross_dataset_comparison.{json,md}`` in *output_dir* and returns
    the test per dimension.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dims = ["fidelity", "utility", "privacy", "overall"]
    frames = []
    for dataset, results in results_by_dataset.items():
        reporter = BenchmarkReporter.__new__(BenchmarkReporter)
        reporter.results = results
        scores = reporter._extract_scores()
        if scores.empty:
            continue
        means = scores.groupby("generator")[dims].mean().reset_index()
        means["dataset"] = dataset
        frames.append(means)
    if not frames:
        return {}
    table = pd.concat(frames, ignore_index=True)
    test = get_statistical_test("friedman-nemenyi", alpha, block_label="dataset")
    datasets = sorted(table["dataset"].unique())
    report: dict = {"datasets": datasets, "statistical_tests": {}}
    lines = [
        "# Cross-dataset comparison (datasets as blocks)",
        "",
        f"Datasets: {', '.join(datasets)}. Score of a generator on a dataset = mean over "
        "its seeds.",
        "",
    ]
    results: dict[str, TestResult] = {}
    for dim in dims:
        pivot = table.pivot(index="dataset", columns="generator", values=dim)
        res = test.compare(pivot, metric_name=dim)
        results[dim] = res
        avg_ranks = pivot.dropna().rank(axis=1, ascending=False).mean().sort_values()
        report["statistical_tests"][dim] = {
            "statistic": None if np.isnan(res.statistic) else float(res.statistic),
            "p_value": None if np.isnan(res.p_value) else float(res.p_value),
            "is_significant": bool(res.is_significant),
            "critical_difference": res.critical_difference,
            "kendall_w": res.kendall_w,
            "n_blocks": res.n_blocks,
            "dropped_blocks": res.dropped_blocks,
            "dropped_methods": res.dropped_methods,
            "average_ranks": avg_ranks.to_dict(),
            "interpretation": res.interpretation,
            "pairwise_comparisons": (
                res.pairwise_comparisons.to_dict(orient="records")
                if res.pairwise_comparisons is not None
                else None
            ),
        }
        lines += [f"## {dim.capitalize()}", "", res.interpretation, ""]
        lines += ["| Generator | Avg rank |", "|---|---|"]
        lines += [f"| {g} | {r:.2f} |" for g, r in avg_ranks.items()]
        lines.append("")
    with (out / "cross_dataset_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
    (out / "cross_dataset_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return results
