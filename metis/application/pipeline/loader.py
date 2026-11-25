"""Step 1 — Load raw data from CSV/Parquet files.

Single responsibility: file I/O and DatasetSpec construction.
"""

from typing import Any

import pandas as pd

from ...domain.entities import DatasetSpec
from ...infrastructure.io.loaders import load_table
from ...infrastructure.runtime.logging import get_logger
from ...shared.config_utils import none_safe as _none_safe


class DataLoader:
    """Loads real and synthetic datasets from disk."""

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    def load(self, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, DatasetSpec]:
        """Load raw DataFrames and build a DatasetSpec from config.

        Args:
            config: Full configuration dict (must contain 'data' section).

        Returns:
            (real_df, synth_df, dataset_spec)
        """
        data_cfg = config["data"]

        # Resolve per-file or global CSV separators
        synth_params = self._csv_params(data_cfg, prefix="synth")

        real_df = self.load_real(config)

        synth_path = _none_safe(data_cfg.get("synthetic"))
        if synth_path is None:
            raise ValueError(
                "'data.synthetic' is required for 'metis evaluate' because "
                "the file-based loader compares real vs synthetic CSV files. "
                "Use a valid synthetic path for evaluate mode."
            )
        synth_df = load_table(synth_path, **synth_params)

        self._logger.info(
            "Loaded raw data: real=%d rows, synth=%d rows",
            len(real_df),
            len(synth_df),
        )

        return real_df, synth_df, self.build_spec(config)

    def load_real(self, config: dict[str, Any]) -> pd.DataFrame:
        """Load only the real table (``data.real``) with its separator."""
        data_cfg = config["data"]
        return load_table(data_cfg["real"], **self._csv_params(data_cfg, prefix="real"))

    def load_holdout(self, config: dict[str, Any]) -> pd.DataFrame | None:
        """Load the optional real holdout (``data.holdout``), or ``None``.

        Separator: ``data.holdout_separator`` → ``data.real_separator`` →
        ``data.separator``; the holdout is real data, so it defaults to the
        real table's format.
        """
        data_cfg = config["data"]
        path = _none_safe(data_cfg.get("holdout"))
        if path is None:
            return None
        holdout = load_table(path, **self._holdout_csv_params(data_cfg))
        self._logger.info("Loaded holdout: %d rows", len(holdout))
        return holdout

    @staticmethod
    def build_spec(config: dict[str, Any]) -> DatasetSpec:
        """DatasetSpec from the ``data`` section."""
        data_cfg = config["data"]
        return DatasetSpec(
            target=_none_safe(data_cfg.get("target")),
            task_type=_none_safe(data_cfg.get("task_type")),
            dtypes=data_cfg.get("dtypes", {}),
            constraints=data_cfg.get("constraints", {}),
        )

    def load_from_dataframes(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, DatasetSpec]:
        """Build DatasetSpec for pre-loaded DataFrames (no file I/O).

        Used by calibration and programmatic API.
        """
        return real_df, synth_df, self.build_spec(config)

    # ----- helpers -----------------------------------------------------------

    @classmethod
    def _holdout_csv_params(cls, data_cfg: dict) -> dict[str, Any]:
        """``holdout_separator`` -> ``real_separator`` -> ``separator``.

        ``_csv_params(prefix="holdout")`` already falls back to the global
        ``separator``, which then won over ``real_separator``.
        """
        if _none_safe(data_cfg.get("holdout_separator")) is not None:
            return cls._csv_params(data_cfg, prefix="holdout")
        return cls._csv_params(data_cfg, prefix="real")

    @staticmethod
    def _csv_params(data_cfg: dict, prefix: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if f"{prefix}_separator" in data_cfg:
            sep = _none_safe(data_cfg[f"{prefix}_separator"])
            if sep is not None:
                params["sep"] = sep
        elif "separator" in data_cfg:
            sep = _none_safe(data_cfg["separator"])
            if sep is not None:
                params["sep"] = sep
        return params
