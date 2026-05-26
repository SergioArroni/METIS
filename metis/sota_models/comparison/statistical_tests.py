"""
Statistical tests for comparing multiple synthetic data generators.

This module implements non-parametric statistical tests to determine if there
are significant differences between generators and identify which pairs differ:

- Friedman test: Non-parametric ANOVA for comparing k methods across multiple datasets
- Nemenyi post-hoc test: Pairwise comparisons after significant Friedman result
- Extensible design allows adding other tests (Wilcoxon, Bonferroni, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    """Results from a statistical significance test."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    pairwise_comparisons: pd.DataFrame | None = None
    critical_difference: float | None = None
    interpretation: str = ""


class StatisticalTest(ABC):
    """
    Abstract base class for statistical tests.

    All statistical tests must inherit from this class and implement
    the compare method. This allows easy extension with new tests.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical test.

        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha

    @abstractmethod
    def compare(self, rankings: pd.DataFrame, metric_name: str = "score") -> TestResult:
        """
        Compare multiple methods using the test.

        Args:
            rankings: DataFrame with shape (n_datasets, n_methods)
                     Each cell contains the rank or score for that method on that dataset
            metric_name: Name of the metric being compared

        Returns:
            TestResult with test statistics and pairwise comparisons
        """
        pass


class FriedmanNemenyiTest(StatisticalTest):
    """
    Friedman test with Nemenyi post-hoc analysis.

    The Friedman test is a non-parametric alternative to repeated measures ANOVA.
    It tests whether k methods have different performance across multiple datasets
    by ranking them on each dataset.

    Process:
    1. Rank methods on each dataset (1=best, k=worst)
    2. Calculate Friedman statistic from average ranks
    3. If significant, perform Nemenyi post-hoc test
    4. Nemenyi identifies which pairs of methods are significantly different

    The Nemenyi test calculates a Critical Difference (CD):
    - If |rank_i - rank_j| > CD, methods i and j are significantly different
    - Based on Studentized range statistic (q_alpha) and number of datasets
    """

    def __init__(self, alpha: float = 0.05, block_label: str = "block"):
        """
        Initialize Friedman-Nemenyi test.

        Args:
            alpha: Significance level (default: 0.05)
            block_label: Human-readable label for the blocking unit (e.g.
                ``"dataset"`` for the canonical use of Friedman, ``"seed"`` for
                repeated runs of a single dataset). This label is propagated
                into ``TestResult.interpretation`` so consumers can audit
                whether the blocks are statistically independent.
        """
        super().__init__(alpha=alpha)
        self.block_label = block_label

    def compare(self, rankings: pd.DataFrame, metric_name: str = "score") -> TestResult:
        """
        Compare methods using Friedman test and Nemenyi post-hoc.

        Args:
            rankings: DataFrame with shape (n_datasets, n_methods)
                     Values should be scores (higher is better)
                     Will be converted to ranks internally
            metric_name: Name of the metric being compared

        Returns:
            TestResult with Friedman statistic, p-value, and pairwise comparisons
        """
        # Convert scores to ranks (1=best)
        # For each row (dataset), rank methods: best gets rank 1
        ranked = rankings.rank(axis=1, ascending=False, method="average")

        n_datasets = len(ranked)
        n_methods = len(ranked.columns)

        if n_datasets < 2:
            return TestResult(
                test_name="Friedman-Nemenyi",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                alpha=self.alpha,
                interpretation="Need at least 2 datasets for Friedman test",
            )

        if n_methods < 2:
            return TestResult(
                test_name="Friedman-Nemenyi",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                alpha=self.alpha,
                interpretation="Need at least 2 methods to compare",
            )

        # Friedman test
        # Convert to format expected by scipy: each column is a method
        statistic, p_value = stats.friedmanchisquare(*[ranked[col] for col in ranked.columns])

        is_significant = p_value < self.alpha

        # Calculate average ranks for each method
        avg_ranks = ranked.mean(axis=0).sort_values()

        # Initialize result
        result = TestResult(
            test_name="Friedman-Nemenyi",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
        )

        # If significant, perform Nemenyi post-hoc test
        small_n_warning = ""
        if n_datasets < 5:
            small_n_warning = (
                f" WARNING: only {n_datasets} {self.block_label}s — Friedman has "
                f"low power below ~5 blocks; treat results as exploratory."
            )

        if is_significant:
            cd, comparisons = self._nemenyi_test(avg_ranks, n_datasets, n_methods)
            result.critical_difference = cd
            result.pairwise_comparisons = comparisons
            result.interpretation = (
                f"Friedman test over {n_datasets} {self.block_label}s is "
                f"significant (p={p_value:.4f} < {self.alpha}). Critical "
                f"Difference (CD) = {cd:.4f}. Pairs with rank difference > CD "
                f"are significantly different.{small_n_warning}"
            )
        else:
            result.interpretation = (
                f"Friedman test over {n_datasets} {self.block_label}s is not "
                f"significant (p={p_value:.4f} >= {self.alpha}). No "
                f"significant differences detected between methods."
                f"{small_n_warning}"
            )

        return result

    def _nemenyi_test(
        self, avg_ranks: pd.Series, n_datasets: int, n_methods: int
    ) -> tuple[float, pd.DataFrame]:
        """
        Perform Nemenyi post-hoc test.

        Args:
            avg_ranks: Average ranks for each method
            n_datasets: Number of datasets (blocks)
            n_methods: Number of methods

        Returns:
            tuple of (critical_difference, pairwise_comparisons_df)
        """
        # CD = q_alpha * sqrt(k(k+1) / (6N))
        # where q_alpha = q_{alpha, k, inf} / sqrt(2) from the studentized range
        # distribution. We compute this exactly via scipy when available.
        q_alpha = self._get_q_alpha(n_methods, self.alpha)
        cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

        # Create pairwise comparison matrix
        methods = avg_ranks.index.tolist()
        comparisons = []

        for i, method_i in enumerate(methods):
            for j, method_j in enumerate(methods):
                if i < j:  # Only upper triangle
                    rank_i = avg_ranks[method_i]
                    rank_j = avg_ranks[method_j]
                    diff = abs(rank_i - rank_j)
                    is_different = diff > cd

                    comparisons.append(
                        {
                            "method_1": method_i,
                            "method_2": method_j,
                            "rank_1": rank_i,
                            "rank_2": rank_j,
                            "rank_diff": diff,
                            "significant": is_different,
                        }
                    )

        comparisons_df = pd.DataFrame(comparisons)

        return cd, comparisons_df

    @staticmethod
    def _get_q_alpha(k: int, alpha: float) -> float:
        """Critical value for the Nemenyi CD.

        Returns ``q_{alpha, k, inf} / sqrt(2)`` from the studentized range
        distribution (the standard Nemenyi formulation in
        Demšar 2006, JMLR 7:1-30). Computed exactly with
        ``scipy.stats.studentized_range`` to avoid the historical
        ``2.0 + 0.2 * log(k)`` approximation.
        """
        if k < 2:
            raise ValueError(f"Nemenyi q_alpha requires k >= 2 (got {k})")
        # studentized_range.ppf returns q_{alpha, k, df=inf}
        q = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf))
        return q / np.sqrt(2.0)


class WilcoxonTest(StatisticalTest):
    """
    Wilcoxon signed-rank test for pairwise comparison.

    This test is for comparing exactly 2 methods. It's a non-parametric
    alternative to paired t-test.

    Use this when you only have 2 methods to compare, or for pairwise
    comparisons with Bonferroni correction.
    """

    def compare(self, rankings: pd.DataFrame, metric_name: str = "score") -> TestResult:
        """
        Compare two methods using Wilcoxon signed-rank test.

        Args:
            rankings: DataFrame with shape (n_datasets, 2)
                     Must have exactly 2 columns (methods)
            metric_name: Name of the metric being compared

        Returns:
            TestResult with Wilcoxon statistic and p-value
        """
        if len(rankings.columns) != 2:
            return TestResult(
                test_name="Wilcoxon",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                alpha=self.alpha,
                interpretation="Wilcoxon test requires exactly 2 methods",
            )

        method1, method2 = rankings.columns
        scores1 = rankings[method1]
        scores2 = rankings[method2]

        diffs = (scores1 - scores2).to_numpy(dtype=float)
        n_nonzero = int(np.sum(diffs != 0))

        # Wilcoxon signed-rank requires at least one non-zero paired difference
        # and is unstable with very few samples. Guard explicitly instead of
        # letting scipy raise (older scipy versions raise ValueError; newer
        # versions emit warnings and may return NaN).
        if n_nonzero < 1:
            return TestResult(
                test_name="Wilcoxon",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                alpha=self.alpha,
                interpretation=(
                    f"Wilcoxon test undefined: all paired differences are zero "
                    f"({method1} vs {method2})."
                ),
            )

        # zero_method='wilcox' (default) drops zero-diff pairs; explicit for
        # reproducibility. mode='exact' is preferable for small samples
        # (n_nonzero < 50) and avoids the normal approximation; scipy auto-
        # selects appropriately when mode='auto'.
        try:
            statistic, p_value = stats.wilcoxon(
                scores1,
                scores2,
                zero_method="wilcox",
                mode="auto",
            )
        except ValueError as exc:
            return TestResult(
                test_name="Wilcoxon",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                alpha=self.alpha,
                interpretation=f"Wilcoxon test failed: {exc}",
            )

        is_significant = p_value < self.alpha

        mean_diff = (scores1 - scores2).mean()

        interpretation = (
            f"Wilcoxon test: {method1} vs {method2}. "
            f"p-value = {p_value:.4f} ({'significant' if is_significant else 'not significant'}). "
            f"Mean difference: {mean_diff:.4f}"
        )

        return TestResult(
            test_name="Wilcoxon",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            interpretation=interpretation,
        )


# Factory function for easy test selection
def get_statistical_test(
    test_name: str = "friedman-nemenyi",
    alpha: float = 0.05,
    block_label: str = "block",
) -> StatisticalTest:
    """
    Factory function to get a statistical test by name.

    Args:
        test_name: Name of the test - "friedman-nemenyi", "wilcoxon"
        alpha: Significance level
        block_label: Optional label propagated to Friedman-family tests so
            that ``TestResult.interpretation`` reports the correct blocking
            unit (e.g. "seed" for repeated runs of a single dataset,
            "dataset" for the canonical multi-dataset benchmark setting).

    Returns:
        StatisticalTest instance

    Raises:
        ValueError: If test_name is unknown
    """
    tests = {
        "friedman-nemenyi": FriedmanNemenyiTest,
        "friedman": FriedmanNemenyiTest,
        "nemenyi": FriedmanNemenyiTest,
        "wilcoxon": WilcoxonTest,
    }

    test_class = tests.get(test_name.lower())
    if test_class is None:
        raise ValueError(f"Unknown test: {test_name}. Available tests: {', '.join(tests.keys())}")

    if test_class is FriedmanNemenyiTest:
        return test_class(alpha=alpha, block_label=block_label)
    return test_class(alpha=alpha)
