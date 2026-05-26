"""Tests for metis.interface.sdk module."""

from unittest.mock import MagicMock, patch

import pandas as pd

from metis.interface.sdk import Evaluator, evaluate_from_config


class TestEvaluator:
    """Tests for the Evaluator SDK class."""

    @patch("metis.interface.sdk.Orchestrator")
    def test_run_from_config_delegates_to_orchestrator(self, mock_orch_cls):
        mock_summary = MagicMock()
        mock_orch_cls.return_value.run.return_value = mock_summary

        evaluator = Evaluator()
        result = evaluator.run_from_config("config.yaml")

        mock_orch_cls.return_value.run.assert_called_once_with("config.yaml")
        assert result is mock_summary

    @patch("metis.interface.sdk.Orchestrator")
    def test_evaluate_delegates_to_orchestrator(self, mock_orch_cls):
        mock_summary = MagicMock()
        mock_orch_cls.return_value.evaluate_dataframes.return_value = mock_summary

        evaluator = Evaluator()
        real = pd.DataFrame({"a": [1, 2]})
        synth = pd.DataFrame({"a": [3, 4]})
        config = {"metrics": ["fidelity.ks"], "reproducibility": {"seed": 123}}

        result = evaluator.evaluate(real, synth, config)

        mock_orch_cls.return_value.evaluate_dataframes.assert_called_once_with(
            real, synth, config, 123
        )
        assert result is mock_summary


class TestEvaluateFromConfig:
    """Tests for the evaluate_from_config convenience function."""

    @patch("metis.interface.sdk.Orchestrator")
    def test_evaluate_from_config_shortcut(self, mock_orch_cls):
        mock_summary = MagicMock()
        mock_orch_cls.return_value.run.return_value = mock_summary

        result = evaluate_from_config("my_config.yaml")

        mock_orch_cls.return_value.run.assert_called_once_with("my_config.yaml")
        assert result is mock_summary
