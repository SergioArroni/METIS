"""Step 1 — Load raw data from CSV/Parquet files.

Single responsibility: file I/O and DatasetSpec construction.
"""

from typing import Any

import pandas as pd

from ...domain.entities import DatasetSpec
from ...infrastructure.io.loaders import load_csv
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
        real_params = self._csv_params(data_cfg, prefix="real")
        synth_params = self._csv_params(data_cfg, prefix="synth")

        real_df = load_csv(data_cfg["real"], **real_params)

        synth_path = _none_safe(data_cfg.get("synthetic"))
        if synth_path is None:
            raise ValueError(
                "'data.synthetic' is required for 'metis evaluate' because "
                "the file-based loader compares real vs synthetic CSV files. "
                "Use a valid synthetic path for evaluate mode."
            )
        synth_df = load_csv(synth_path, **synth_params)

        self._logger.info(
            "Loaded raw data: real=%d rows, synth=%d rows",
            len(real_df),
            len(synth_df),
        )

        spec = DatasetSpec(
            target=_none_safe(data_cfg.get("target")),
            task_type=_none_safe(data_cfg.get("task_type")),
            dtypes=data_cfg.get("dtypes", {}),
            constraints=data_cfg.get("constraints", {}),
        )

        return real_df, synth_df, spec

    def load_from_dataframes(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, DatasetSpec]:
        """Build DatasetSpec for pre-loaded DataFrames (no file I/O).

        Used by calibration and programmatic API.
        """
        data_cfg = config["data"]
        spec = DatasetSpec(
            target=_none_safe(data_cfg.get("target")),
            task_type=_none_safe(data_cfg.get("task_type")),
            dtypes=data_cfg.get("dtypes", {}),
            constraints=data_cfg.get("constraints", {}),
        )
        return real_df, synth_df, spec

    # ----- helpers -----------------------------------------------------------

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
