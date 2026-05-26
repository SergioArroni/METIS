"""
Multi-run statistics calculation for reproducibility experiments.

This module provides functions to aggregate results across multiple runs
and calculate relevant statistics (median, mean, quantiles, etc.).
"""

from typing import Any

import numpy as np
from scipy import stats as scipy_stats


def calculate_multi_run_statistics(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate statistics across multiple runs for all score levels.

    Args:
        run_results: list of summary dictionaries from each run

    Returns:
        Dictionary with aggregated statistics including:
        - median, mean, std for all scores
        - quantiles (Q1, Q3)
        - min, max
        - mode (if applicable)

    Example:
        >>> run_results = [
        ...     {"scores": {"fidelity_score": 0.72, "composite_score": 0.68}},
        ...     {"scores": {"fidelity_score": 0.75, "composite_score": 0.70}},
        ...     {"scores": {"fidelity_score": 0.73, "composite_score": 0.69}},
        ... ]
        >>> stats = calculate_multi_run_statistics(run_results)
        >>> print(stats["fidelity_score"]["median"])
        0.73
    """
    if not run_results:
        return {}

    # Extract all score keys from first run
    first_scores = run_results[0].get("scores", {})
    score_keys = [k for k in first_scores if k.endswith("_score")]

    statistics = {}

    for key in score_keys:
        # Collect values across all runs
        values = []
        for run_result in run_results:
            score_value = run_result.get("scores", {}).get(key)
            if score_value is not None:
                values.append(score_value)

        if not values:
            continue

        values_array = np.array(values)

        # Calculate statistics
        stats_dict = {
            "median": float(np.median(values_array)),
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "q1": float(np.percentile(values_array, 25)),
            "q3": float(np.percentile(values_array, 75)),
            "n_runs": len(values),
        }

        # Calculate mode if there are repeated values
        if len(values) > 1:
            try:
                mode_result = scipy_stats.mode(values_array, keepdims=True)
                # Only include mode if it appears more than once
                if mode_result.count[0] > 1:
                    stats_dict["mode"] = float(mode_result.mode[0])
            except Exception:
                pass  # Skip mode if calculation fails

        statistics[key] = stats_dict

    # Also aggregate hierarchy_breakdown statistics if present
    if "hierarchy_breakdown" in run_results[0]:
        statistics["hierarchy_stats"] = _aggregate_hierarchy_stats(run_results)

    return statistics


def _aggregate_hierarchy_stats(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate statistics for hierarchical breakdown across runs.

    Args:
        run_results: list of summary dictionaries from each run

    Returns:
        Aggregated hierarchy statistics
    """
    hierarchy_stats = {}

    # Get structure from first run
    first_hierarchy = run_results[0].get("hierarchy_breakdown", {})

    for domain_name, domain_data in first_hierarchy.items():
        domain_stats = {}

        # Collect domain scores
        domain_scores = []
        for run_result in run_results:
            score = run_result.get("hierarchy_breakdown", {}).get(domain_name, {}).get("score")
            if score is not None:
                domain_scores.append(score)

        if domain_scores:
            domain_stats["score"] = _calculate_score_stats(domain_scores)

        # Aggregate category-level stats
        if "categories" in domain_data:
            category_stats = {}
            for cat_name, cat_data in domain_data["categories"].items():
                cat_scores = []
                for run_result in run_results:
                    cat_score = (
                        run_result.get("hierarchy_breakdown", {})
                        .get(domain_name, {})
                        .get("categories", {})
                        .get(cat_name, {})
                        .get("score")
                    )
                    if cat_score is not None:
                        cat_scores.append(cat_score)

                if cat_scores:
                    category_stats[cat_name] = {"score": _calculate_score_stats(cat_scores)}

                # Aggregate subcategory stats if present
                if "subcategories" in cat_data:
                    subcat_stats = {}
                    for subcat_name, _subcat_data in cat_data["subcategories"].items():
                        subcat_scores = []
                        for run_result in run_results:
                            subcat_score = (
                                run_result.get("hierarchy_breakdown", {})
                                .get(domain_name, {})
                                .get("categories", {})
                                .get(cat_name, {})
                                .get("subcategories", {})
                                .get(subcat_name, {})
                                .get("score")
                            )
                            if subcat_score is not None:
                                subcat_scores.append(subcat_score)

                        if subcat_scores:
                            subcat_stats[subcat_name] = _calculate_score_stats(subcat_scores)

                    if subcat_stats:
                        category_stats[cat_name]["subcategories"] = subcat_stats

            if category_stats:
                domain_stats["categories"] = category_stats

        if domain_stats:
            hierarchy_stats[domain_name] = domain_stats

    return hierarchy_stats


def _calculate_score_stats(scores: list[float]) -> dict[str, float]:
    """Helper to calculate statistics for a list of scores."""
    scores_array = np.array(scores)

    return {
        "median": float(np.median(scores_array)),
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array, ddof=1)) if len(scores) > 1 else 0.0,
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "q1": float(np.percentile(scores_array, 25)),
        "q3": float(np.percentile(scores_array, 75)),
    }
