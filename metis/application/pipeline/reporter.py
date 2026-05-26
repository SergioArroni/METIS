"""Step 7 — Render and persist evaluation reports.

Single responsibility: take a RunSummary + config and write report files
(JSON, Markdown, etc.) using the Reporter registry.
"""

import json
from pathlib import Path
from typing import Any

from ...domain.entities import ReportSpec, RunSummary
from ...infrastructure.reporting.registry import get_reporter_registry
from ...infrastructure.runtime.logging import get_logger


class ReportGenerator:
    """Generates reports in all configured formats."""

    def __init__(self) -> None:
        self._logger = get_logger(__name__)
        self._reporter_registry = get_reporter_registry()

    def generate(self, summary: RunSummary, config: dict[str, Any]) -> None:
        """Write reports to disk according to config['report'].

        Silently skips if no report section is present or formats list is empty.
        """
        report_cfg = config.get("report")
        if not report_cfg:
            return

        formats = report_cfg.get("formats", [])
        if not formats:
            return

        spec = ReportSpec(
            formats=formats,
            output_dir=report_cfg["output_dir"],
            include_details=report_cfg.get("include_details", True),
            include_artifacts=report_cfg.get("include_artifacts", False),
        )
        Path(spec.output_dir).mkdir(parents=True, exist_ok=True)

        for fmt in spec.formats:
            try:
                reporter_cls = self._reporter_registry.get(fmt)
                reporter_cls().render(summary, spec)
                self._logger.info("Generated %s report → %s", fmt, spec.output_dir)
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Failed to generate %s report: %s", fmt, exc)

    # ----- helpers for multi-run persistence ---------------------------------

    def save_run_json(self, summary: RunSummary, output_path: Path) -> None:
        """Persist a single run's summary dict to a JSON file."""
        from ...infrastructure.reporting.json_reporter import JSONReporter

        temp_spec = ReportSpec(
            formats=["json"],
            output_dir=str(output_path.parent),
            include_details=False,
            include_artifacts=False,
        )
        data = JSONReporter().serialize_summary(summary, temp_spec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def extract_scores(self, summary: RunSummary) -> dict[str, Any]:
        """Extract score/hierarchy dicts for multi-run statistics."""
        from ...infrastructure.reporting.json_reporter import JSONReporter

        temp_spec = ReportSpec(
            formats=["json"],
            output_dir=".",
            include_details=False,
            include_artifacts=False,
        )
        data = JSONReporter().serialize_summary(summary, temp_spec)
        return {
            "scores": data.get("scores", {}),
            "hierarchy_breakdown": data.get("hierarchy_breakdown", {}),
        }
