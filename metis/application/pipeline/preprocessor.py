"""Step 2 — Preprocess raw data: type casting, cat/num split.

Single responsibility: schema-based transformation.

Missing values are **kept** (they used to be removed row-wise: any row with
a NaN in any column was dropped from the real, synthetic and holdout tables
before every metric, so a table with gaps spread over many columns was
evaluated on a small, biased subset and the synthetic gaps were invisible).
Each metric uses the values it has (per column, per pair, or with an
explicit missingness encoding).

Values the caster cannot read (an ordinal level that does not exist, an
unparsable number, a discrete value outside its ranges, an unknown boolean)
become missing as well, in every view of the column, and are counted per
table in ``meta["_rows"]`` (``cells_invalid``, ``rows_with_invalid``,
``invalid_by_column``): they are not dropped silently, and the rules axis
checks every synthetic value against the declared schema.

The real table is the *reference*: the caster is fitted on it, and every
other table (synthetic, holdout) is transformed with that same fitted caster.
:class:`ReferenceContext` keeps that reference so programmatic callers that
compare many synthetic tables against the same real one pay its
preprocessing once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ...domain.contracts import TypeSchema
from ...domain.entities import DatasetSpec, TransformedData
from ...domain.errors import ConfigError, TypeCastingError
from ...infrastructure.preprocess.caster import SimpleCaster
from ...infrastructure.runtime.logging import get_logger
from ...shared.schema_utils import EXCLUDED_SCHEMA_TYPES, filter_schema_columns


@dataclass
class ReferenceContext:
    """Preprocessed real (reference) table, reusable across evaluations.

    Holds everything that depends only on the real table and the ``data``
    section of the config: the caster fitted on the real rows, the
    transformed real table and its row accounting.

    ``real_clean`` is the real table *before* casting (ID columns stripped;
    missing values are kept).  Calibration needs that one: its strategies
    evaluate real halves and noise through the whole pipeline again, so a
    table that is already cast would be cast twice (a datetime column
    renamed to ``<col>__ts`` vanished from the schema, a discrete column
    already turned into bin positions was re-binned into one bin).  Build it once with
    :meth:`DataPreprocessor.prepare_reference` and pass it to
    :meth:`DataPreprocessor.transform_other` for every synthetic (or holdout)
    table compared against the same reference.
    """

    caster: SimpleCaster
    real: TransformedData
    schema_cfg: dict[str, Any]
    real_stats: dict[str, int] = field(default_factory=dict)
    real_clean: pd.DataFrame | None = None

    def fresh_real(self) -> TransformedData:
        """Copy of the transformed real table, so a metric cannot alter the cache."""
        td = self.real
        return TransformedData(
            cat=td.cat.copy(),
            num=td.num.copy(),
            full=td.full.copy(),
            meta=dict(td.meta),
            excluded_ids=list(td.excluded_ids),
            schema=td.schema,
        )


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
        """Full preprocessing pipeline: strip IDs → type casting.

        Steps:
          0. Strip ID columns (the only excluded type).
          1. Build TypeSchema from config (excluding IDs).
          2. Fit the caster on the real rows and transform both tables.
          3. Values the cast could not read become missing (counted).

        No row is dropped.  The accounting of each table (``rows_in``,
        ``rows_with_missing``, ``cells_missing``, ``rows_with_invalid``,
        ``cells_invalid``, ``invalid_by_column``, ``rows_out``; the former
        ``rows_dropped_nan``/``rows_dropped_cast`` stay at 0) is kept in
        ``TransformedData.meta["_rows"]``.

        Returns:
            (real_transformed, synth_transformed)
        """
        ref = self.prepare_reference(real, config, spec)
        synth_tf = self.transform_other(synth, ref, label="synth")
        return ref.real, synth_tf

    def prepare_reference(
        self,
        real: pd.DataFrame,
        config: dict[str, Any],
        spec: DatasetSpec | None = None,
    ) -> ReferenceContext:
        """Steps 0–4 of :meth:`preprocess` for the real table only."""
        schema_cfg = config["data"].get("schema")
        if schema_cfg == "auto":  # the Phase −1 rules on the real table
            from metis.infrastructure.typing import infer_schema

            schema_cfg = infer_schema(real).schema
        if not schema_cfg:
            raise ConfigError(
                "Missing required 'schema' section in 'data' config. "
                "You must define the semantic type for each column."
            )
        schema_cfg = self._normalise_schema_types(schema_cfg)
        self._logger.info("Starting preprocessing with schema: %s", schema_cfg)

        type_schema = TypeSchema(columns=filter_schema_columns(schema_cfg))
        self._check_schema_columns(type_schema, real, "real")

        real_clean = self._strip_ids(real, schema_cfg)

        caster = SimpleCaster(type_schema)
        caster.fit(real_clean, spec)
        real_tf, warnings = caster.transform_to_entity(real_clean, "real")
        for warning in warnings:
            self._logger.warning(warning)
        real_tf, invalid = self._mark_invalid(real_tf, real_clean, type_schema, "real")
        stats = self._row_stats(real_clean, type_schema, invalid)
        real_tf.meta["_rows"] = stats

        # The caster only sees the evaluable columns, so the stripped ID
        # columns are recorded here (the artifact was always empty before).
        real_tf.excluded_ids = self._id_columns(schema_cfg, real)
        if real_tf.excluded_ids:
            self._logger.info("Excluded ID columns: %s", real_tf.excluded_ids)
        return ReferenceContext(
            caster=caster,
            real=real_tf,
            schema_cfg=schema_cfg,
            real_stats=stats,
            real_clean=real_clean,
        )

    def transform_other(
        self,
        df: pd.DataFrame,
        ref: ReferenceContext,
        label: str = "synth",
    ) -> TransformedData:
        """Preprocess a synthetic or holdout table exactly like the real one.

        Same ID stripping and the caster *fitted on the real rows* (never
        refitted on *df*).
        """
        self._check_schema_columns(ref.caster.schema, df, label)
        clean = self._strip_ids(df, ref.schema_cfg)
        tf, warnings = ref.caster.transform_to_entity(clean, label)
        for warning in warnings:
            self._logger.warning(warning)
        tf, invalid = self._mark_invalid(tf, clean, ref.caster.schema, label)
        tf.meta["_rows"] = self._row_stats(clean, ref.caster.schema, invalid)
        self._logger.info("Transformed %s data: cat=%s, num=%s", label, tf.cat.shape, tf.num.shape)
        return tf

    # ----- helpers -----------------------------------------------------------

    @staticmethod
    def _normalise_schema_types(schema_cfg: dict[str, Any]) -> dict[str, Any]:
        """Schema with every type written as ``TypeSchema`` reads it (``"ID"`` → ``"id"``).

        ``TypeSchema`` strips and lower-cases the type, but the ID stripping
        compared the raw string: ``type: ID`` was neither stripped nor
        excluded, and the identifier was evaluated as a numeric column.
        """
        out: dict[str, Any] = {}
        for col, spec in schema_cfg.items():
            if isinstance(spec, str):
                out[col] = spec.strip().lower()
            elif isinstance(spec, dict) and isinstance(spec.get("type"), str):
                out[col] = {**spec, "type": spec["type"].strip().lower()}
            else:
                out[col] = spec
        return out

    @staticmethod
    def _id_columns(schema_cfg: dict[str, Any], df: pd.DataFrame) -> list[str]:
        return [
            col
            for col, spec in schema_cfg.items()
            if (spec if isinstance(spec, str) else spec.get("type", "")) in EXCLUDED_SCHEMA_TYPES
            and col in df.columns
        ]

    @staticmethod
    def _row_stats(
        raw: pd.DataFrame, type_schema: TypeSchema, invalid: pd.DataFrame
    ) -> dict[str, Any]:
        """Missing and invalid values of one table (no row is dropped any more)."""
        cols = [c for c in type_schema.columns if c in raw.columns]
        missing = raw[cols].isna() if cols else pd.DataFrame(index=raw.index)
        return {
            "rows_in": int(len(raw)),
            "rows_with_missing": int(missing.any(axis=1).sum()) if cols else 0,
            "cells_missing": int(missing.to_numpy().sum()) if cols else 0,
            "rows_with_invalid": int(invalid.any(axis=1).sum()) if not invalid.empty else 0,
            "cells_invalid": int(invalid.to_numpy().sum()) if not invalid.empty else 0,
            "invalid_by_column": (
                {c: int(n) for c, n in invalid.sum().items() if n} if not invalid.empty else {}
            ),
            # Kept for readers of the former accounting: nothing is dropped.
            "rows_dropped_nan": 0,
            "rows_dropped_cast": 0,
            "rows_out": int(len(raw)),
        }

    @staticmethod
    def _check_schema_columns(type_schema: TypeSchema, df: pd.DataFrame, label: str) -> None:
        missing = set(type_schema.columns) - set(df.columns)
        if missing:
            name = {"real": "real", "synth": "synthetic"}.get(label, label)
            raise TypeCastingError(f"Schema columns missing in {name} data: {sorted(missing)}")

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
        to_drop = self._id_columns(schema_cfg, df)
        if to_drop:
            self._logger.info("Stripping ID columns: %s", to_drop)
            df = df.drop(columns=to_drop)
        else:
            self._logger.info("No ID columns to strip.")
        # A fresh 0..n-1 index: the cast and every metric align by position.
        return df.reset_index(drop=True)

    def _mark_invalid(
        self, td: TransformedData, raw: pd.DataFrame, type_schema: TypeSchema, label: str
    ) -> tuple[TransformedData, pd.DataFrame]:
        """Values the cast could not read become missing in every view of their column.

        A transformed cell is *invalid* when it is NaN but the raw value was
        present (an ordinal level that does not exist, an unparsable number, a
        discrete value outside its ranges, an unknown boolean).  Such rows used
        to be dropped; now the cell is a gap, counted in the returned frame
        (one boolean column per source column).  An ordinal's categorical view
        keeps the raw string after the cast, so it is blanked too: both views
        must agree on which values exist.
        """
        cat, num = td.cat.copy(), td.num.copy()
        invalid: dict[str, pd.Series] = {}
        for frame in (cat, num):
            for col in frame.columns:
                src = self._source_column(col, type_schema)
                if src is None or src not in raw.columns:
                    continue
                bad = frame[col].isna().to_numpy() & raw[src].notna().to_numpy()
                if bad.any():
                    invalid[src] = invalid.get(src, pd.Series(False, index=raw.index)) | bad
        for src, bad in invalid.items():
            for frame in (cat, num):
                for col in frame.columns:
                    if self._source_column(col, type_schema) == src:
                        frame.loc[bad.to_numpy(), col] = None if frame is cat else float("nan")
        invalid_df = pd.DataFrame(invalid, index=raw.index)
        if invalid:
            counts = {c: int(v.sum()) for c, v in invalid.items()}
            self._logger.warning(
                "%s data: %d value(s) could not be read and are treated as missing: %s",
                label,
                sum(counts.values()),
                counts,
            )
        return (
            TransformedData(
                cat=cat,
                num=num,
                full=pd.concat([cat, num], axis=1),
                meta=td.meta,
                excluded_ids=td.excluded_ids,
                schema=td.schema,
            ),
            invalid_df,
        )

    @staticmethod
    def _source_column(col: str, type_schema: TypeSchema) -> str | None:
        """Raw column a transformed column comes from (``x__ord``/``x__ts`` → ``x``)."""
        if col in type_schema.columns:
            return col
        for suffix in ("__ord", "__ts"):
            if col.endswith(suffix) and col[: -len(suffix)] in type_schema.columns:
                return col[: -len(suffix)]
        return None
