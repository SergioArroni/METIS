"""Domain taxonomy defining metric families, categories, and evaluation metrics."""

# =============================================================================
# METRIC FAMILIES
# =============================================================================
FAMILIES: set[str] = {"fidelity", "utility", "privacy"}

# =============================================================================
# METRIC TAXONOMY - Hierarchical organization of all available metrics
# =============================================================================

# Fidelity Metrics - Measure statistical similarity between real and synthetic data
FIDELITY_METRICS: dict[str, dict[str, list[str]]] = {
    "global": {
        "description": "Structural metrics comparing overall distributions",
        "metrics": [
            "fidelity.correlation_matrix",
            "fidelity.mmd",
            "fidelity.energy_distance",
            "fidelity.outliers_coverage",
        ],
    },
    "marginal": {
        "description": "Per-column distribution comparison",
        "subcategories": {
            "tails": [
                "fidelity.ks",  # Kolmogorov-Smirnov
                "fidelity.wasserstein",
                "fidelity.anderson_darling",
                "fidelity.hellinger",
                "fidelity.kde_ise",
                "fidelity.delta_exceedance",
            ],
            "scales": [
                "fidelity.delta_mean",
                "fidelity.delta_median",
                "fidelity.delta_iqr",
                "fidelity.delta_mad",
                "fidelity.cohens_d",
            ],
            "coverage": [
                "fidelity.tvd",  # Total Variation Distance
                "fidelity.js",  # Jensen-Shannon
                "fidelity.kl",  # Kullback-Leibler
                "fidelity.psi",  # Population Stability Index
                "fidelity.entropy_delta",
                "fidelity.gini_delta",
            ],
        },
    },
    "conditional": {
        "description": "Bivariate relationship comparison",
        "subcategories": {
            "num_num": [
                "fidelity.pearson",
                "fidelity.spearman",
                "fidelity.dcor",  # Distance correlation
                "fidelity.mi",  # Mutual Information
            ],
            "num_cat": [
                "fidelity.eta_squared",
                "fidelity.point_biserial",
                "fidelity.kruskal_epsilon",
            ],
            "cat_cat": [
                "fidelity.cramers_v",
                "fidelity.theils_u",
                "fidelity.chi2_stat",
            ],
        },
    },
}

# Privacy Metrics - Measure privacy preservation
PRIVACY_METRICS: dict[str, dict[str, list[str]]] = {
    "dataset_based": {
        "description": "Privacy metrics based on dataset analysis",
        "subcategories": {
            "attribute_inference": [
                "privacy.mia",  # Membership Inference Attack
                "privacy.inference_attack",  # General Attribute Inference
            ],
            "reidentification": [
                "privacy.k_anonymity",
                "privacy.l_diversity",
                "privacy.t_closeness",
                "privacy.record_linkage",
            ],
            "empirical_similarity": [
                "privacy.dcr",  # Distance to Closest Record
                "privacy.nnaa",  # Nearest Neighbor Adversarial Accuracy
            ],
        },
    },
    "mechanism_based": {
        "description": "Privacy metrics based on generation mechanism",
        "metrics": [
            "privacy.differential_privacy",
        ],
    },
}

# Utility Metrics - Measure usefulness for downstream tasks
UTILITY_METRICS: dict[str, dict[str, list[str]]] = {
    "ml_efficiency": {
        "description": "ML efficiency evaluation with auto task detection and configurable strategies",
        "subcategories": {
            "standalone": [
                "utility.tts",
                "utility.tstr",
                "utility.trts",
                "utility.ttrs",
            ],
            "aggregate": [
                "utility.ml_efficiency",
            ],
        },
    },
}

# =============================================================================
# METRIC HIERARCHY LOOKUP
# =============================================================================


def _build_metric_hierarchy_map() -> dict[str, dict[str, str]]:
    """Build a lookup map from metric_id to its hierarchy (family, category, subcategory)."""
    hierarchy_map: dict[str, dict[str, str]] = {}

    all_taxonomies = [
        ("fidelity", FIDELITY_METRICS),
        ("privacy", PRIVACY_METRICS),
        ("utility", UTILITY_METRICS),
    ]

    for family, taxonomy in all_taxonomies:
        for category, category_data in taxonomy.items():
            if "metrics" in category_data:
                for metric_id in category_data["metrics"]:
                    hierarchy_map[metric_id] = {
                        "family": family,
                        "category": category,
                        "subcategory": None,
                    }
            if "subcategories" in category_data:
                for subcategory, metrics in category_data["subcategories"].items():
                    for metric_id in metrics:
                        hierarchy_map[metric_id] = {
                            "family": family,
                            "category": category,
                            "subcategory": subcategory,
                        }

    return hierarchy_map


# Pre-built hierarchy map for fast lookup
_METRIC_HIERARCHY_MAP: dict[str, dict[str, str]] = _build_metric_hierarchy_map()


def get_metric_hierarchy(metric_id: str) -> dict[str, str]:
    """
    Get the full hierarchy path for a metric.

    Args:
        metric_id: The metric ID (e.g., "fidelity.ks" or just "ks")

    Returns:
        Dictionary with keys: family, category, subcategory (subcategory may be None)

    Example:
        >>> get_metric_hierarchy("fidelity.ks")
        {"family": "fidelity", "category": "marginal", "subcategory": "tails"}
        >>> get_metric_hierarchy("ks")
        {"family": "fidelity", "category": "marginal", "subcategory": "tails"}
    """
    # Direct lookup with full ID
    if metric_id in _METRIC_HIERARCHY_MAP:
        return _METRIC_HIERARCHY_MAP[metric_id].copy()

    # Try to find by short name (without family prefix)
    # This handles cases where metric_id is just "ks" instead of "fidelity.ks"
    for full_id, hierarchy in _METRIC_HIERARCHY_MAP.items():
        # Extract the short name from the full ID (e.g., "ks" from "fidelity.ks")
        if "." in full_id:
            short_name = full_id.split(".")[-1]
            if short_name == metric_id:
                return hierarchy.copy()

    # Fallback: try to infer from metric_id prefix
    if "." in metric_id:
        family = metric_id.split(".")[0]
        if family in FAMILIES:
            return {"family": family, "category": "unknown", "subcategory": None}

    return {"family": "unknown", "category": "unknown", "subcategory": None}


# =============================================================================
# METRIC ID EXPANSION — shorthand resolution for config files
# =============================================================================

_ALL_TAXONOMY: dict[str, dict] = {
    "fidelity": FIDELITY_METRICS,
    "privacy": PRIVACY_METRICS,
    "utility": UTILITY_METRICS,
}


def _extract_metrics_from_category(cat_data: dict) -> list[str]:
    """Extract all metric IDs from a category dict."""
    metrics: list[str] = []
    if "metrics" in cat_data:
        metrics.extend(cat_data["metrics"])
    if "subcategories" in cat_data:
        for subcat_metrics in cat_data["subcategories"].values():
            metrics.extend(subcat_metrics)
    return metrics


def expand_metric_ids(raw_ids: list[str]) -> list[str]:
    """Expand category / subcategory shortcuts into individual metric IDs.

    Accepted shorthand formats:
        - ``"fidelity"``               → every fidelity metric
        - ``"fidelity.marginal"``       → every metric in the *marginal* category
        - ``"fidelity.marginal.tails"`` → every metric in the *tails* subcategory
        - ``"fidelity.ks"``            → kept as-is (already a concrete metric)

    Unknown entries are passed through unchanged so that the registry
    can emit its own error later.
    """
    expanded: list[str] = []
    seen: set[str] = set()

    def _add(mid: str) -> None:
        if mid not in seen:
            expanded.append(mid)
            seen.add(mid)

    def _add_all(mids: list[str]) -> None:
        for mid in mids:
            _add(mid)

    for entry in raw_ids:
        # Already a concrete metric?
        if entry in _METRIC_HIERARCHY_MAP:
            _add(entry)
            continue

        resolved = _resolve_shorthand(entry)
        _add_all(resolved)

    return expanded


def _resolve_shorthand(entry: str) -> list[str]:
    """Resolve a single shorthand entry into metric IDs."""
    parts = entry.split(".")

    if len(parts) == 1:
        return _resolve_family(parts[0], entry)
    if len(parts) == 2:
        return _resolve_category(parts[0], parts[1], entry)
    if len(parts) == 3:
        return _resolve_subcategory(parts[0], parts[1], parts[2], entry)
    return [entry]


def _resolve_family(family: str, fallback: str) -> list[str]:
    if family not in _ALL_TAXONOMY:
        return [fallback]
    result: list[str] = []
    for cat_data in _ALL_TAXONOMY[family].values():
        result.extend(_extract_metrics_from_category(cat_data))
    return result


def _resolve_category(family: str, category: str, fallback: str) -> list[str]:
    tax = _ALL_TAXONOMY.get(family, {})
    if category not in tax:
        return [fallback]
    return _extract_metrics_from_category(tax[category])


def _resolve_subcategory(family: str, category: str, subcategory: str, fallback: str) -> list[str]:
    subcats = _ALL_TAXONOMY.get(family, {}).get(category, {}).get("subcategories", {})
    if subcategory not in subcats:
        return [fallback]
    return list(subcats[subcategory])


def group_metrics_by_hierarchy(
    metric_ids: list[str],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """
    Group metric IDs by their hierarchy (family -> category -> subcategory).

    Args:
        metric_ids: list of metric IDs

    Returns:
        Nested dictionary: {family: {category: {subcategory: [metric_ids]}}}
        For metrics without subcategory, subcategory key will be "_direct"
    """
    grouped: dict[str, dict[str, dict[str, list[str]]]] = {}

    for metric_id in metric_ids:
        hierarchy = get_metric_hierarchy(metric_id)
        family = hierarchy["family"]
        category = hierarchy["category"]
        subcategory = hierarchy["subcategory"] or "_direct"

        if family not in grouped:
            grouped[family] = {}
        if category not in grouped[family]:
            grouped[family][category] = {}
        if subcategory not in grouped[family][category]:
            grouped[family][category][subcategory] = []

        grouped[family][category][subcategory].append(metric_id)

    return grouped
