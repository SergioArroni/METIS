"""Tests for metis.infrastructure.reporting module."""

import json
from pathlib import Path

import pytest

from metis.domain.entities import EvalPlan, MetricResult, ReportSpec, RunSummary
from metis.infrastructure.reporting.json_reporter import JSONReporter
from metis.infrastructure.reporting.markdown_reporter import MarkdownReporter


@pytest.fixture
def run_summary():
    """Create a RunSummary for testing reporters."""
    plan = EvalPlan(metric_ids=["fidelity.ks", "fidelity.wasserstein"], seed=42, cv_splits=3)
    results = [
        MetricResult(id="fidelity.ks", value=0.85, details={"p_value": 0.3}, family="fidelity"),
        MetricResult(id="fidelity.wasserstein", value=0.72, details={}, family="fidelity"),
    ]
    return RunSummary(
        plan=plan,
        results=results,
        aggregates={
            "fidelity_score": 0.78,
            "composite_score": 0.78,
            "total_metrics": 2,
            "successful_metrics": 2,
            "failed_metrics": 0,
        },
        artifacts={"seed": 42},
    )


@pytest.fixture
def report_spec(tmp_path):
    """Create a ReportSpec pointing to tmp directory."""
    return ReportSpec(
        output_dir=str(tmp_path / "reports"),
        formats=["json", "md"],
        include_artifacts=False,
    )


class TestJSONReporter:
    """Tests for JSONReporter."""

    def test_render_creates_summary_json(self, run_summary, report_spec, tmp_path):
        reporter = JSONReporter()
        reporter.render(run_summary, report_spec)

        summary_path = Path(report_spec.output_dir) / "summary.json"
        assert summary_path.exists()

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "metadata" in data
        assert "scores" in data
        assert "metrics" in data
        assert data["metadata"]["generated_by"] == "METIS"

    def test_render_creates_all_metrics_json(self, run_summary, report_spec):
        reporter = JSONReporter()
        reporter.render(run_summary, report_spec)

        all_metrics_path = Path(report_spec.output_dir) / "all_metrics.json"
        assert all_metrics_path.exists()

        data = json.loads(all_metrics_path.read_text(encoding="utf-8"))
        assert "plan" in data
        assert "results" in data
        assert len(data["results"]) == 2

    def test_serialize_summary_structure(self, run_summary, report_spec):
        reporter = JSONReporter()
        summary = reporter.serialize_summary(run_summary, report_spec)

        assert summary["summary"]["total_metrics"] == 2
        assert summary["summary"]["successful_metrics"] == 2
        assert summary["summary"]["failed_metrics"] == 0
        assert "scores" in summary

    def test_metrics_include_id_and_value(self, run_summary, report_spec):
        reporter = JSONReporter()
        summary = reporter.serialize_summary(run_summary, report_spec)

        metrics = summary["metrics"]
        assert len(metrics) == 2
        assert metrics[0]["id"] == "fidelity.ks"
        assert metrics[0]["value"] == 0.85


class TestMarkdownReporter:
    """Tests for MarkdownReporter."""

    def test_render_creates_summary_md(self, run_summary, report_spec):
        reporter = MarkdownReporter()
        reporter.render(run_summary, report_spec)

        md_path = Path(report_spec.output_dir) / "summary.md"
        assert md_path.exists()

        content = md_path.read_text(encoding="utf-8")
        assert "METIS" in content
        assert "Evaluation Report" in content

    def test_report_contains_metric_count(self, run_summary, report_spec):
        reporter = MarkdownReporter()
        reporter.render(run_summary, report_spec)

        md_path = Path(report_spec.output_dir) / "summary.md"
        content = md_path.read_text(encoding="utf-8")
        assert "2" in content  # total metrics count
