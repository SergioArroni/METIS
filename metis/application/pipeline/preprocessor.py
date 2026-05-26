"""Step 2 — Preprocess raw data: NaN removal, type casting, cat/num split.

Single responsibility: data cleaning and schema-based transformation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...domain.contracts import TypeSchema
from ...domain.entities import DatasetSpec, TransformedData
from ...domain.errors import ConfigError
from ...infrastructure.io.schema import align_schema
from ...infrastructure.runtime.logging import get_logger
from ...shared.schema_utils import EXCLUDED_SCHEMA_TYPES, filter_schema_columns


class DataPreprocessor:
    """Cleans raw DataFrames and transforms them into typed cat/num splits."""

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    def preprocess(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
        config: dict[str, Any],
        spec: DatasetSpec | None = None,
    ) -> tuple[TransformedData, TransformedData]:
        """Full preprocessing pipeline: strip IDs → NaN removal → type casting.

        Steps:
          0. Strip ID columns (the only excluded type).
          1. Drop rows containing NaN.
          2. Build TypeSchema from config (excluding IDs).
          3. Call ``align_schema`` to split cat / num columns.

        Returns:
            (real_transformed, synth_transformed)
        """
        # 0. Strip ID columns
        schema_cfg = config["data"].get("schema")
        if not schema_cfg:
            raise ConfigError(
                "Missing required 'schema' section in 'data' config. "
                "You must define the semantic type for each column."
            )

        self._logger.info("Starting preprocessing with schema: %s", schema_cfg)

        real = self._strip_ids(real, schema_cfg)
        synth = self._strip_ids(synth, schema_cfg)

        self._logger.info(
            "After ID stripping: real=%d columns, synth=%d columns",
            real.shape[1],
            synth.shape[1],
        )

        # 1. Remove NaN rows
        real_clean = self._drop_nan(real, label="real")
        synth_clean = self._drop_nan(synth, label="synth")

        self._logger.info(
            "After NaN removal: real=%d, synth=%d rows",
            len(real_clean),
            len(synth_clean),
        )

        # 2. Build TypeSchema (only evaluable columns, no IDs)
        evaluable_schema = filter_schema_columns(schema_cfg)
        type_schema = TypeSchema(columns=evaluable_schema)

        self._logger.info(
            "Constructed TypeSchema with %d columns: %s",
            len(type_schema.columns),
            list(type_schema.columns.keys()),
        )

        # 3. Align & transform (uses SimpleCaster internally)
        real_tf, synth_tf, _ = align_schema(real_clean, synth_clean, spec, type_schema)

        # 4. Post-transformation NaN cleanup
        #    type casting (e.g. ordinal levels not in synth) can introduce NaN.
        real_tf = self._clean_transformed(real_tf, "real")
        synth_tf = self._clean_transformed(synth_tf, "synth")

        self._logger.info(
            "Transformed data: real_cat=%s, real_num=%s, synth_cat=%s, synth_num=%s",
            real_tf.cat.shape,
            real_tf.num.shape,
            synth_tf.cat.shape,
            synth_tf.num.shape,
        )
        if real_tf.excluded_ids:
            self._logger.info("Excluded ID columns: %s", real_tf.excluded_ids)

        return real_tf, synth_tf

    # ----- helpers -----------------------------------------------------------

    def _strip_ids(
        self,
        df: pd.DataFrame,
        schema_cfg: dict[str, Any],
    ) -> pd.DataFrame:
        """Remove columns typed as ``id`` from a DataFrame.

        Called as step 0 of :meth:`preprocess` so that all downstream
        transformations and calibration fingerprints operate on the
        same column set.
        """
        excluded = [
            col
            for col, spec in schema_cfg.items()
            if (spec if isinstance(spec, str) else spec.get("type", "")) in EXCLUDED_SCHEMA_TYPES
        ]
        to_drop = [c for c in excluded if c in df.columns]
        if to_drop:
            self._logger.info("Stripping ID columns: %s", to_drop)
            return df.drop(columns=to_drop)
        self._logger.info("No ID columns to strip.")
        return df

    def _drop_nan(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        before = len(df)
        clean = df.dropna().reset_index(drop=True)
        dropped = before - len(clean)
        if dropped:
            self._logger.warning(
                "Dropped %d rows (%.1f%%) from %s data due to NaN values",
                dropped,
                100 * dropped / before,
                label,
            )
            return clean
        self._logger.info("No NaN values found in %s data.", label)
        return clean

    def _clean_transformed(self, td: TransformedData, label: str) -> TransformedData:
        """Drop rows where type casting introduced NaN and rebuild full."""
        self._logger.debug(
            "Checking for NaN in transformed %s data: cat=%s, num=%s",
            label,
            td.cat.shape,
            td.num.shape,
        )

        cat_nans = td.cat.isna().any(axis=1)
        num_nans = td.num.isna().any(axis=1)
        bad_rows = cat_nans | num_nans

        if not bad_rows.any():
            self._logger.info(
                "No NaN detected after type casting in %s data — all rows valid", label
            )
            return td

        keep = ~bad_rows
        dropped = int(bad_rows.sum())
        cat_dropped = int(cat_nans.sum())
        num_dropped = int(num_nans.sum())

        self._logger.warning(
            "Dropped %d rows (%.1f%%) from %s data — NaN introduced during type casting "
            "(cat: %d rows, num: %d rows)",
            dropped,
            100 * dropped / len(td.cat),
            label,
            cat_dropped,
            num_dropped,
        )

        new_cat = td.cat.loc[keep].reset_index(drop=True)
        new_num = td.num.loc[keep].reset_index(drop=True)
        new_full = pd.concat([new_cat, new_num], axis=1)

        self._logger.info(
            "Rebuilt %s TransformedData: cat=%s, num=%s",
            label,
            new_cat.shape,
            new_num.shape,
        )

        return TransformedData(
            cat=new_cat,
            num=new_num,
            full=new_full,
            meta=td.meta,
            excluded_ids=td.excluded_ids,
            schema=td.schema,
        )
