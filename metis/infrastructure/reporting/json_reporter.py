"""JSON reporter for structured evaluation results."""

import json
from pathlib import Path
from typing import Any

from ...domain.entities import ReportSpec, RunSummary
from ...domain.taxonomy import get_metric_hierarchy


class JSONReporter:
    """Reporter that generates JSON output for structured data consumption."""

    def render(self, run_summary: RunSummary, report_spec: ReportSpec) -> None:
        """Generate JSON reports from run summary.

        Generates two files:
        - summary.json: Compact summary with scores and hierarchy
        - all_metrics.json: Full detailed information for all metrics
        """
        output_dir = Path(report_spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate compact summary
        summary_data = self.serialize_summary(run_summary, report_spec)
        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

        # Generate full metrics details
        all_metrics_data = self._serialize_all_metrics(run_summary, report_spec)
        all_metrics_path = output_dir / "all_metrics.json"
        with all_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(all_metrics_data, f, indent=2, ensure_ascii=False, default=str)

    def serialize_summary(
        self, run_summary: RunSummary, _report_spec: ReportSpec
    ) -> dict[str, Any]:
        """Create compact summary without metric details."""

        # Build metrics summary (id, value, category only)
        metrics_summary = []
        for result in run_summary.results:
            hierarchy = get_metric_hierarchy(result.id)
            metrics_summary.append(
                {
                    "id": result.id,
                    "value": result.value,
                    "category": hierarchy.get("category", "unknown"),
                    "subcategory": hierarchy.get("subcategory"),
                }
            )

        # Aggregates - check if multi-run stats are available
        has_multi_run = run_summary.artifacts and "multi_run_stats" in run_summary.artifacts

        if has_multi_run:
            # Use multi-run statistics in scores section
            multi_run_stats = run_summary.artifacts["multi_run_stats"]
            aggregates_data = {}

            # Keep non-score fields as is
            for key, value in run_summary.aggregates.items():
                if (
                    key != "hierarchy"
                    and not key.endswith("_score")
                    and not key.endswith("_count")
                    and key
                    not in (
                        "composite_families",
                        "total_metrics",
                        "successful_metrics",
                        "failed_metrics",
                    )
                ):
                    aggregates_data[key] = value

            # Replace score values with multi-run statistics
            for key, value in run_summary.aggregates.items():
                if key.endswith("_score") and key in multi_run_stats:
                    # Replace scalar with statistics dict
                    aggregates_data[key] = multi_run_stats[key]
                elif key in (
                    "composite_families",
                    "total_metrics",
                    "successful_metrics",
                    "failed_metrics",
                ) or key.endswith("_count"):
                    aggregates_data[key] = value
        else:
            # Single run: use original aggregates
            aggregates_data = {}
            for key, value in run_summary.aggregates.items():
                if key != "hierarchy":
                    aggregates_data[key] = value

        summary = {
            "metadata": {
                "report_format": "json",
                "generated_by": "METIS",
                "version": "0.1.0",
            },
            "summary": {
                "total_metrics": len(run_summary.results),
                "successful_metrics": len(
                    [r for r in run_summary.results if "error" not in r.details]
                ),
                "failed_metrics": len([r for r in run_summary.results if "error" in r.details]),
            },
            "scores": aggregates_data,
            "metrics": metrics_summary,
            "hierarchy_breakdown": (
                run_summary.aggregates.get("hierarchy")
                or self._build_hierarchy_breakdown(run_summary)
            ),
        }

        # Add multi-run statistics if available
        if run_summary.artifacts and "multi_run_stats" in run_summary.artifacts:
            summary["multi_run_statistics"] = run_summary.artifacts["multi_run_stats"]
            summary["reproducibility"] = {
                "n_runs": run_summary.artifacts.get("n_runs"),
                "base_seed": run_summary.artifacts.get("base_seed"),
                "seeds_used": run_summary.artifacts.get("seeds_used"),
            }

        # Add schema summary
        if run_summary.artifacts:
            schema_applied = run_summary.artifacts.get("schema_applied", {})
            excluded_ids = run_summary.artifacts.get("excluded_id_columns", [])
            summary["data_schema"] = {
                "total_columns": len(schema_applied) + len(excluded_ids),
                "numeric_columns": len(
                    [
                        c
                        for c, t in schema_applied.items()
                        if t in ("continuous", "discrete", "ordinal", "datetime")
                    ]
                ),
                "categorical_columns": len(
                    [
                        c
                        for c, t in schema_applied.items()
                        if t in ("categorical", "boolean", "text", "code_numeric")
                    ]
                ),
            }

        return summary

    def _serialize_all_metrics(
        self, run_summary: RunSummary, report_spec: ReportSpec
    ) -> dict[str, Any]:
        """Create full metrics report with all details."""

        # Serialize evaluation plan
        plan_data = {
            "metric_ids": run_summary.plan.metric_ids,
            "seed": run_summary.plan.seed,
            "cv_splits": run_summary.plan.cv_splits,
        }

        # Serialize results with full details
        results_data = []
        for result in run_summary.results:
            hierarchy = get_metric_hierarchy(result.id)
            result_data = {
                "id": result.id,
                "value": result.value,
                "family": result.family,
                "category": hierarchy.get("category", "unknown"),
                "subcategory": hierarchy.get("subcategory"),
                "purpose_tags": list(result.purpose_tags),
                "details": result.details,  # Always include full details
            }
            results_data.append(result_data)

        # Full schema information
        data_schema = {}
        if run_summary.artifacts:
            schema_applied = run_summary.artifacts.get("schema_applied", {})
            excluded_ids = run_summary.artifacts.get("excluded_id_columns", [])
            data_schema = {
                "column_types": schema_applied,
                "excluded_id_columns": excluded_ids,
                "total_columns": len(schema_applied) + len(excluded_ids),
                "numeric_columns": len(
                    [
                        c
                        for c, t in schema_applied.items()
                        if t in ("continuous", "discrete", "ordinal", "datetime")
                    ]
                ),
                "categorical_columns": len(
                    [
                        c
                        for c, t in schema_applied.items()
                        if t in ("categorical", "boolean", "text", "code_numeric")
                    ]
                ),
            }

        all_metrics = {
            "metadata": {
                "report_format": "json",
                "generated_by": "METIS",
                "version": "0.1.0",
                "report_type": "all_metrics",
            },
            "plan": plan_data,
            "results": results_data,
            "data_schema": data_schema,
        }

        # Add artifacts if requested
        if report_spec.include_artifacts and run_summary.artifacts:
            all_metrics["artifacts"] = run_summary.artifacts

        return all_metrics

    def _build_hierarchy_breakdown(self, run_summary: RunSummary) -> dict[str, Any]:
        """
        Build hierarchical breakdown from results when not available in aggregates.

        This is a fallback for backwards compatibility.
        """
        import statistics

        hierarchy: dict[str, Any] = {}

        for result in run_summary.results:
            metric_hierarchy = get_metric_hierarchy(result.id)
            family = metric_hierarchy["family"]
            category = metric_hierarchy["category"]
            subcategory = metric_hierarchy["subcategory"]

            # Initialize family structure
            if family not in hierarchy:
                hierarchy[family] = {"categories": {}, "score": 0.0, "count": 0}

            # Initialize category structure
            if category not in hierarchy[family]["categories"]:
                hierarchy[family]["categories"][category] = {"score": 0.0, "count": 0}

            cat_data = hierarchy[family]["categories"][category]

            if subcategory:
                # Initialize subcategories dict
                if "subcategories" not in cat_data:
                    cat_data["subcategories"] = {}

                if subcategory not in cat_data["subcategories"]:
                    cat_data["subcategories"][subcategory] = {
                        "score": 0.0,
                        "count": 0,
                        "metrics": [],
                        "_values": [],
                    }

                subcat_data = cat_data["subcategories"][subcategory]
                metric_name = result.id.split(".")[-1] if "." in result.id else result.id
                subcat_data["metrics"].append(metric_name)
                subcat_data["_values"].append(result.value)
            else:
                if "metrics" not in cat_data:
                    cat_data["metrics"] = []
                    cat_data["_values"] = []
                metric_name = result.id.split(".")[-1] if "." in result.id else result.id
                cat_data["metrics"].append(metric_name)
                cat_data["_values"].append(result.value)

        # Calculate scores at each level
        for _family, family_data in hierarchy.items():
            all_family_values = []

            for _category, cat_data in family_data["categories"].items():
                cat_values = []

                # Process subcategories
                if "subcategories" in cat_data:
                    for _subcat, subcat_data in cat_data["subcategories"].items():
                        if subcat_data["_values"]:
                            subcat_data["score"] = statistics.mean(subcat_data["_values"])
                            subcat_data["count"] = len(subcat_data["_values"])
                            cat_values.extend(subcat_data["_values"])
                        del subcat_data["_values"]  # Clean up temporary data

                # Process direct metrics
                if "_values" in cat_data:
                    cat_values.extend(cat_data["_values"])
                    del cat_data["_values"]

                if cat_values:
                    cat_data["score"] = statistics.mean(cat_values)
                    cat_data["count"] = len(cat_values)
                    all_family_values.extend(cat_values)

            if all_family_values:
                family_data["score"] = statistics.mean(all_family_values)
                family_data["count"] = len(all_family_values)

        return hierarchy
