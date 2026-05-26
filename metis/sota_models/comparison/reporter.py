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

DISPLAY_ANCHOR_DIMENSIONS = {
    "real_data": {"fidelity": 1.0, "utility": 1.0, "privacy": 0.0},
    "uniform_noise": {"fidelity": 0.0, "utility": 0.0, "privacy": 1.0},
}

# Airbnb currently uses a mean composite after the cache refresh, so the
# displayed anchor rows should recompute the overall score from the displayed
# family extremes to keep tables and figure aligned.
DISPLAY_MEAN_COMPOSITE_OUTPUT_DIRS = {"benchmark_airbnb"}


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

        display_scores_df = self._adjust_display_scores(scores_df)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(display_scores_df)

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

        for gen_name, gen_results in self.results.items():
            for seed, seed_results in gen_results.items():
                if "error" not in seed_results:
                    # Support both formats: new format (aggregated_scores) and old (run_summary)
                    if "aggregated_scores" in seed_results:
                        aggregates = seed_results["aggregated_scores"]
                        row = {
                            "generator": gen_name,
                            "seed": seed,
                            "fidelity": aggregates.get("fidelity", 0.0),
                            "utility": aggregates.get("utility", 0.0),
                            "privacy": aggregates.get("privacy", 0.0),
                            "overall": aggregates.get("overall", 0.0),
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
                            "fidelity": aggregates.get("fidelity_score", 0.0),
                            "utility": aggregates.get("utility_score", 0.0),
                            "privacy": aggregates.get("privacy_score", 0.0),
                            "overall": aggregates.get("composite_score", 0.0),
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

            row = {"generator": gen_name}

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

    def _adjust_display_scores(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Clamp calibration anchors for presentation without altering raw exports."""
        adjusted = scores_df.copy()

        if "generator" not in adjusted.columns:
            return adjusted

        for generator, dimensions in DISPLAY_ANCHOR_DIMENSIONS.items():
            generator_mask = adjusted["generator"] == generator
            if not generator_mask.any():
                continue

            for dimension, value in dimensions.items():
                if dimension in adjusted.columns:
                    adjusted.loc[generator_mask, dimension] = value

        if self.output_dir.name in DISPLAY_MEAN_COMPOSITE_OUTPUT_DIRS:
            anchor_mask = adjusted["generator"].isin(DISPLAY_ANCHOR_DIMENSIONS.keys())
            if anchor_mask.any() and {"fidelity", "utility", "privacy"}.issubset(adjusted.columns):
                adjusted.loc[anchor_mask, "overall"] = adjusted.loc[
                    anchor_mask, ["fidelity", "utility", "privacy"]
                ].mean(axis=1)

        return adjusted

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
            f.write("## Statistical Significance Tests\n\n")

            for dim in ["fidelity", "utility", "privacy", "overall"]:
                result = test_results[dim]

                f.write(f"### {dim.capitalize()}\n\n")
                f.write(f"**Test**: {result.test_name}\n\n")
                f.write(f"**Statistic**: {result.statistic:.4f}\n\n")
                f.write(f"**p-value**: {result.p_value:.4f}\n\n")
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
            avg_ranks.to_csv(self.output_dir / f"rankings_{dim}.csv", header=["average_rank"])
