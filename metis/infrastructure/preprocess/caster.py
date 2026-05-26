"""
SimpleCaster - Heterogeneous data type transformation.

Transforms data with heterogeneous types into uniform categorical (CAT)
and numeric (NUM) DataFrames for metric computation.

Supported types:
    - boolean: Boolean → CAT ('yes'/'no')
    - categorical: Categorical → CAT (normalized string)
    - ordinal: Ordered categorical → NUM [0,1]
    - continuous: Continuous numeric → NUM (float)
    - discrete: Discrete numeric with ranges → NUM [0,1] normalized by bin
    - datetime: DateTime → NUM (timestamp seconds)
    - geospatial: Geographic → NUM (lat/lon) or CAT
    - text: Text → CAT (with optional top-k mapping)
    - code_numeric: Numeric codes → CAT (string)
    - id: Identifier → EXCLUDED from output
"""

from typing import Any, Self

import numpy as np
import pandas as pd

from metis.domain.contracts import TypeSchema
from metis.domain.entities import DatasetSpec, TransformedData
from metis.domain.errors import TypeCastingError


class SimpleCaster:
    """
    Transform heterogeneous data types to uniform CAT/NUM DataFrames.

    Implements the Preprocessor protocol with fit/transform pattern.
    Columns declared as 'id' are excluded from output.

    Usage:
        schema = TypeSchema(columns={
            'age': 'continuous',
            'gender': 'categorical',
            'education': {'type': 'ordinal', 'levels': ['low', 'medium', 'high']},
            'patient_id': 'id'
        })
        caster = SimpleCaster(schema)
        caster.fit(real_df)

        real_cat, real_num = caster.transform(real_df)
        synth_cat, synth_num = caster.transform(synth_df)
    """

    _TEXT_TOPK = 100_000
    """Threshold for top-k mapping in text columns."""

    # ------------------------------------------------------------------ #
    #  Private helpers (static)                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_str(x) -> str | None:
        """Normalize string: strip whitespace and lowercase."""
        if pd.isna(x):
            return x
        return str(x).strip().lower()

    @staticmethod
    def _is_num(s: pd.Series) -> bool:
        """Check if series has numeric dtype."""
        return s.dtype.kind in ("i", "u", "f")

    @staticmethod
    def _top_k_map(series: pd.Series, k: int) -> dict | None:
        """
        Create top-K mapping for high-cardinality categorical columns.

        Returns None if cardinality <= k, otherwise returns mapping dict
        where values not in top-k are mapped to 'OTHER'.
        """
        vc = series.value_counts(dropna=False)
        if len(vc) <= k:
            return None
        keep = set(vc.head(k).index)
        return {val: (val if val in keep else "OTHER") for val in vc.index}

    # ------------------------------------------------------------------ #
    #  Validation (static)                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_schema_columns(
        schema: TypeSchema, real_df: pd.DataFrame, synth_df: pd.DataFrame
    ) -> None:
        """
        Validate that schema columns exist in both datasets.

        Args:
            schema: TypeSchema with column definitions
            real_df: Real DataFrame
            synth_df: Synthetic DataFrame

        Raises:
            TypeCastingError: If required columns are missing
        """
        schema_cols = set(schema.columns.keys())

        missing_in_real = schema_cols - set(real_df.columns)
        if missing_in_real:
            raise TypeCastingError(
                f"Schema columns missing in real data: {sorted(missing_in_real)}"
            )

        missing_in_synth = schema_cols - set(synth_df.columns)
        if missing_in_synth:
            raise TypeCastingError(
                f"Schema columns missing in synthetic data: {sorted(missing_in_synth)}"
            )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, schema: TypeSchema):
        """
        Initialize SimpleCaster with type schema.

        Args:
            schema: TypeSchema defining column types
        """
        self.schema = schema
        self._is_fitted = False
        self._meta: dict[str, Any] = {}
        self._excluded_ids: list[str] = []

    def fit(self, data: pd.DataFrame, spec: DatasetSpec | None = None) -> Self:
        """
        Fit caster to real data.

        Learns column statistics and mappings from real data.
        These are then applied consistently to both real and synthetic data.

        Args:
            data: Real DataFrame to fit on
            spec: Optional DatasetSpec (for protocol compatibility)

        Returns:
            Self for method chaining
        """
        meta: dict[str, Any] = {}

        for col, col_spec in self.schema.columns.items():
            typ = col_spec.type

            # Track ID columns
            if typ == "id":
                self._excluded_ids.append(col)
                meta[col] = {
                    "type": typ,
                    "present": col in data.columns,
                    "excluded": True,
                }
                continue

            if col not in data.columns:
                meta[col] = {"type": typ, "present": False}
                continue

            s = data[col]
            m: dict[str, Any] = {"type": typ, "present": True}

            # ORDINAL: levels come from schema (validated in ColumnTypeSpec)
            if typ == "ordinal":
                m["levels"] = list(col_spec.levels) if col_spec.levels else []

            # DISCRETE: ranges come from schema (validated in ColumnTypeSpec)
            if typ == "discrete":
                m["ranges"] = list(col_spec.ranges) if col_spec.ranges else []

            # TEXT: compute top-k mapping if cardinality > threshold
            if typ == "text":
                norm = s.astype("object").map(self._normalize_str)
                mp = self._top_k_map(norm, k=self._TEXT_TOPK)
                if mp:
                    m["top_k_map"] = mp

            meta[col] = m

        # Check for geospatial lat/lon columns
        meta["_geo_has_latlon"] = ("lat" in data.columns) and ("lon" in data.columns)

        self._meta = meta
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform DataFrame to CAT and NUM DataFrames.

        Each semantic type is delegated to a registered handler in
        :pyattr:`_TYPE_HANDLERS` (Strategy + dispatch table). Adding a new
        type means writing a ``_cast_<name>`` method and listing it in the
        registry — no edits to this method needed (OCP).

        Args:
            df: DataFrame to transform

        Returns:
            tuple of (df_cat, df_num) DataFrames
        """
        if not self._is_fitted:
            raise TypeCastingError("SimpleCaster must be fitted before transform")

        df_cat: dict[str, pd.Series] = {}
        df_num: dict[str, pd.Series] = {}

        geo_ok = self._meta.get("_geo_has_latlon", False)
        ctx = {
            "geo_ok": geo_ok,
            "lat_series": df["lat"] if geo_ok and "lat" in df.columns else None,
            "lon_series": df["lon"] if geo_ok and "lon" in df.columns else None,
        }

        for col, col_spec in self.schema.columns.items():
            if col not in df.columns:
                continue

            s = df[col]
            typ = col_spec.type

            try:
                handler_name = self._TYPE_HANDLERS.get(typ)
                if handler_name is not None:
                    getattr(self, handler_name)(col, s, col_spec, df_cat, df_num, ctx)
                elif self._is_num(s):
                    df_num[col] = s.astype("float")
                else:
                    df_cat[col] = s.astype("object").map(self._normalize_str)

            except Exception as e:
                raise TypeCastingError(
                    "Failed to cast column",
                    column=col,
                    expected_type=typ,
                    original_error=e,
                ) from e

        return pd.DataFrame(df_cat), pd.DataFrame(df_num)

    # ------------------------------------------------------------------ #
    #  Type handlers (Strategy registry below)                            #
    # ------------------------------------------------------------------ #

    def _cast_code_numeric(self, col, s, _spec, df_cat, _df_num, _ctx):
        df_cat[col] = s.astype("object").map(lambda v: None if pd.isna(v) else str(v))

    def _cast_boolean(self, col, s, _spec, df_cat, _df_num, _ctx):
        sv = s.astype("object").map(lambda v: str(v).lower() if not pd.isna(v) else v)
        df_cat[col] = sv.map(
            lambda v: (
                "yes"
                if v in {"1", "true", "yes", "sí", "si"}
                else ("no" if v in {"0", "false", "no"} else np.nan)
            )
        )

    def _cast_categorical(self, col, s, _spec, df_cat, _df_num, _ctx):
        df_cat[col] = s.astype("object").map(self._normalize_str)

    def _cast_ordinal(self, col, s, col_spec, df_cat, df_num, _ctx):
        norm_s = s.astype("object").map(self._normalize_str)
        df_cat[col] = norm_s

        levels = col_spec.levels or []
        idx = {self._normalize_str(v): i + 1 for i, v in enumerate(levels)} if levels else {}
        k = max(1, len(levels))

        num = norm_s.map(idx).astype("float")
        if k > 1:
            num01 = (num - 1.0) / float(k - 1)
        else:
            num01 = num.map(lambda r: 0.0 if pd.notna(r) else np.nan)
        df_num[col + "__ord"] = num01.astype("float")

    def _cast_continuous(self, col, s, _spec, _df_cat, df_num, _ctx):
        s_clean = s.astype(str).str.replace(",", ".", regex=False)
        df_num[col] = pd.to_numeric(s_clean, errors="coerce").astype("float")

    def _cast_discrete(self, col, s, col_spec, _df_cat, df_num, _ctx):
        ranges = col_spec.ranges or []

        def map_to_bin(v, rngs=ranges):
            try:
                if pd.isna(v):
                    return np.nan
                fv = float(v)
            except (ValueError, TypeError):
                return np.nan
            for i, (lo, hi) in enumerate(rngs, start=1):
                if lo <= fv <= hi:
                    return i
            return np.nan

        bin_idx = s.map(map_to_bin)
        k = max(1, len(ranges))
        if k > 1:
            num01 = (bin_idx.astype("float") - 1.0) / float(k - 1)
        else:
            num01 = bin_idx.map(lambda r: 0.0 if pd.notna(r) else np.nan)
        df_num[col] = num01.astype("float")

    def _cast_datetime(self, col, s, _spec, _df_cat, df_num, _ctx):
        ts = pd.to_datetime(s, errors="coerce")
        df_num[col + "__ts"] = (ts.astype("int64") / 1e9).astype("float")

    def _cast_geospatial(self, col, s, _spec, df_cat, df_num, ctx):
        if ctx["geo_ok"] and ctx["lat_series"] is not None and ctx["lon_series"] is not None:
            df_num["lat"] = ctx["lat_series"].astype("float")
            df_num["lon"] = ctx["lon_series"].astype("float")
        else:
            df_cat[col] = s.astype("object").map(self._normalize_str)

    def _cast_text(self, col, s, _spec, df_cat, _df_num, _ctx):
        x = s.astype("object").map(self._normalize_str)
        mp = self._meta.get(col, {}).get("top_k_map")
        if mp:
            x = x.map(lambda v, top_k_map=mp: top_k_map.get(v, "OTHER"))
        df_cat[col] = x

    # Type → handler-method-name registry. Add new semantic types here
    # without touching :py:meth:`transform`.
    _TYPE_HANDLERS: dict[str, str] = {
        "code_numeric": "_cast_code_numeric",
        "boolean": "_cast_boolean",
        "categorical": "_cast_categorical",
        "ordinal": "_cast_ordinal",
        "continuous": "_cast_continuous",
        "discrete": "_cast_discrete",
        "datetime": "_cast_datetime",
        "geospatial": "_cast_geospatial",
        "text": "_cast_text",
    }

    def transform_to_entity(
        self, df: pd.DataFrame, dataset_name: str = "data"
    ) -> tuple[TransformedData, list[str]]:
        """
        Transform DataFrame and return as TransformedData entity.

        Returns warnings for categories defined in schema but not present in data.
        NaN handling should be done on raw data BEFORE calling this method.

        Args:
            df: DataFrame to transform (should be NaN-free)
            dataset_name: Name for warning messages ('real' or 'synth')

        Returns:
            tuple of (TransformedData, list of warning messages)
        """
        warnings = []

        for col, col_spec in self.schema.columns.items():
            if col not in df.columns:
                continue

            if col_spec.type == "ordinal" and col_spec.levels:
                present_values = set(df[col].dropna().unique())
                expected_levels = set(col_spec.levels)
                missing = expected_levels - present_values
                if missing:
                    warnings.append(
                        f"[{dataset_name}] Column '{col}': categories {sorted(missing)} "
                        f"not present in data (has {sorted(present_values)})"
                    )

        df_cat, df_num = self.transform(df)
        df_full = pd.concat([df_cat, df_num], axis=1)

        transformed = TransformedData(
            cat=df_cat,
            num=df_num,
            full=df_full,
            meta=self._meta.copy(),
            excluded_ids=self._excluded_ids.copy(),
            schema=self.schema,
        )

        return transformed, warnings

    def metadata(self) -> dict[str, Any]:
        """Return transformation metadata."""
        if not self._is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "meta": self._meta,
            "excluded_ids": self._excluded_ids,
            "n_columns": len(self.schema.columns),
            "n_excluded": len(self._excluded_ids),
        }

    def get_excluded_ids(self) -> list[str]:
        """Get list of columns declared as 'id' that were excluded."""
        return self._excluded_ids.copy()
