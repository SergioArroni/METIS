"""Tests for metis.interface.cli module."""

from unittest.mock import MagicMock, patch

from metis.interface.cli import _build_parser, main


class TestBuildParser:
    """Tests for CLI parser construction."""

    def test_parser_has_evaluate_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["evaluate", "--config", "test.yaml"])
        assert args.command == "evaluate"
        assert args.config == "test.yaml"

    def test_parser_has_calibrate_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["calibrate", "--config", "test.yaml"])
        assert args.command == "calibrate"
        assert args.config == "test.yaml"

    def test_parser_calibrate_iterations_option(self):
        parser = _build_parser()
        args = parser.parse_args(["calibrate", "-c", "test.yaml", "-n", "20"])
        assert args.iterations == 20

    def test_parser_calibrate_iterations_default_none(self):
        parser = _build_parser()
        args = parser.parse_args(["calibrate", "-c", "test.yaml"])
        assert args.iterations is None

    def test_parser_has_version_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["version"])
        assert args.command == "version"

    def test_parser_no_command_returns_none(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestMainCommand:
    """Tests for main() dispatch."""

    def test_version_prints_version(self, capsys):
        result = main(["version"])
        captured = capsys.readouterr()
        assert result == 0
        assert "METIS" in captured.out

    def test_no_command_prints_help(self, capsys):
        result = main([])
        captured = capsys.readouterr()
        assert result == 0
        assert "usage" in captured.out.lower() or "METIS" in captured.out

    @patch("metis.application.orchestrator.Orchestrator")
    def test_evaluate_success(self, mock_orch_cls, capsys):
        mock_summary = MagicMock()
        mock_summary.aggregates = {"composite_score": 0.85}
        mock_orch_cls.return_value.run.return_value = mock_summary

        result = main(["evaluate", "--config", "test.yaml"])
        assert result == 0
        captured = capsys.readouterr()
        assert "0.85" in captured.out

    @patch("metis.application.orchestrator.Orchestrator")
    def test_evaluate_error_returns_1(self, mock_orch_cls, capsys):
        mock_orch_cls.return_value.run.side_effect = RuntimeError("boom")

        result = main(["evaluate", "--config", "test.yaml"])
        assert result == 1
        captured = capsys.readouterr()
        assert "boom" in captured.err

    @patch("metis.calibrate.cache.cache_manager.CacheManager")
    @patch("metis.infrastructure.io.loaders.load_csv")
    @patch("metis.infrastructure.runtime.config.load_config")
    def test_calibrate_success(self, mock_load_config, mock_load_csv, mock_cache, capsys):
        mock_load_config.return_value = {
            "data": {"real": "test.csv"},
            "calibration": {"n_iterations": 5},
        }
        mock_load_csv.return_value = MagicMock()
        mock_bounds = MagicMock()
        mock_bounds.get_summary.return_value = "OK"
        mock_cache.return_value.get_or_calibrate.return_value = mock_bounds

        result = main(["calibrate", "--config", "test.yaml"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Calibration complete" in captured.out

    @patch("metis.infrastructure.runtime.config.load_config")
    def test_calibrate_error_returns_1(self, mock_load_config, capsys):
        mock_load_config.side_effect = FileNotFoundError("not found")

        result = main(["calibrate", "--config", "missing.yaml"])
        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err
