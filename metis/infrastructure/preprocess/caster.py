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
    - geospatial: Geographic → NUM when the real column is numeric (a latitude,
      a longitude…), CAT otherwise (geohash, place names)
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
            'children': {'type': 'discrete', 'levels': [0, 1, 2, 3, 4]},
            'gender': 'nominal',
            'education': {'type': 'ordinal', 'levels': ['low', 'medium', 'high']},
            'patient_id': 'id'
        })
        caster = SimpleCaster(schema)
        caster.fit(real_df)

        real_cat, real_num = caster.transform(real_df)
        synth_cat, synth_num = caster.transform(synth_df)
    """

    # ------------------------------------------------------------------ #
    #  Private helpers (static)                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonical_str(x) -> str:
        """String form of a scalar where ``1``, ``1.0`` and ``np.int64(1)`` agree.

        Integer-valued floats are written without the decimal part, so a real
        column read as ``int`` and a synthetic one returned as ``float`` land
        on the same category (``"1"``) instead of ``"1"`` vs ``"1.0"``.
        """
        if isinstance(x, bool | np.bool_):
            return "true" if x else "false"
        if isinstance(x, float | np.floating) and np.isfinite(x) and float(x).is_integer():
            return str(int(x))
        if isinstance(x, np.integer):
            return str(int(x))
        return str(x)

    @staticmethod
    def _normalize_str(x) -> str | None:
        """Normalize string: strip whitespace and lowercase."""
        if pd.isna(x):
            return x
        return SimpleCaster._canonical_str(x).strip().lower()

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
        self._excluded_ids = []  # refitting must not accumulate duplicates

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

            # CONTINUOUS: dates and timestamps are measured in seconds; declared
            # or recognised on the real column (non-numeric values that parse
            # as dates), so real and synthetic are read the same way.
            if typ == "continuous":
                m["datetime"] = bool(col_spec.datetime) or self._is_datetime_like(s)

            meta[col] = m

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

        ctx: dict[str, Any] = {}

        for col, col_spec in self.schema.columns.items():
            if col not in df.columns:
                continue

            s = df[col]
            typ = col_spec.type
            if typ == "id":  # identifiers are never evaluated
                continue

            try:
                getattr(self, self._TYPE_HANDLERS[typ])(col, s, col_spec, df_cat, df_num, ctx)

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

    def _cast_nominal(self, col, s, _spec, df_cat, _df_num, _ctx):
        """Categories without order: the value as a normalised string (``1``,
        ``1.0`` and ``" 1 "`` are one category; booleans read ``true``/``false``)."""
        df_cat[col] = s.astype("object").map(self._normalize_str)

    def _cast_ordinal(self, col, s, col_spec, df_cat, df_num, _ctx):
        """An ordinal in two views, used the same way everywhere in METIS.

        ``df_cat[col]`` holds the level and ``df_num[col + "__ord"]`` its rank
        ``i/(K-1)`` in [0, 1] (``i`` the 0-based position in ``levels``;
        an unknown level has no rank).  Every computation takes the view that
        matches what it measures:

        - category frequencies and contingency (TVD, JS, Cramér's V, χ², the
          quasi-identifiers of k-anonymity/l-diversity/t-closeness): the level;
        - distance or order (KS/Wasserstein on the num slice, Spearman/Pearson,
          MMD, energy, the rank against nominal columns in num-cat pairs, and
          the record distances of DCR/NNAA/MIA/record linkage, unscaled:
          Gower's ordinal distance): the rank;
        - models (CatBoost in utility, the attacker of the inference attack):
          both; an ordinal *target* is a class scored with quadratic weighted
          kappa on the ranks.

        Within one metric each pair of variables is counted once.
        Decision and evidence: ``docs/auditoria/2026-10-09-decisiones-de-metodo.md``.
        """
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
        """Any value within the range.  Dates and timestamps become seconds
        since the epoch in ``<col>__ts`` (``NaT`` stays NaN)."""
        if self._meta.get(col, {}).get("datetime", False):
            ts = pd.to_datetime(s, errors="coerce")
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            seconds = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
            df_num[col + "__ts"] = seconds.astype("float")
            return
        s_clean = s.astype(str).str.replace(",", ".", regex=False)
        df_num[col] = pd.to_numeric(s_clean, errors="coerce").astype("float")

    def _cast_discrete(self, col, s, _spec, _df_cat, df_num, _ctx):
        """A discrete numeric is read by its value.

        Whether the value belongs to its declared domain (2.5 children) is a
        domain rule, checked by the constraints axis
        (``metis.infrastructure.constraints``) and reported apart; fidelity
        compares the distributions as they come.  (Until 2026-10-20 a
        categorical shadow ``<col>__lvl`` brought that check into the
        marginal metrics: the same defect counted twice.)
        """
        s_clean = s.astype(str).str.replace(",", ".", regex=False)
        df_num[col] = pd.to_numeric(s_clean, errors="coerce").astype("float")

    # Type → handler-method-name registry (the four types of Phase −1).
    _TYPE_HANDLERS: dict[str, str] = {
        "nominal": "_cast_nominal",
        "ordinal": "_cast_ordinal",
        "continuous": "_cast_continuous",
        "discrete": "_cast_discrete",
    }

    # ------------------------------------------------------------------ #
    #  Read on the real column                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_datetime_like(s: pd.Series) -> bool:
        """Non-numeric values of which at least 90 % parse as dates."""
        non_null = s.dropna()
        if non_null.empty or s.dtype.kind in ("i", "u", "f"):
            return False
        as_text = non_null.astype(str)
        if pd.to_numeric(as_text.str.replace(",", ".", regex=False), errors="coerce").notna().all():
            return False
        parsed = pd.to_datetime(as_text, errors="coerce", format="mixed")
        return bool(parsed.notna().mean() >= 0.9)

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
                # Compared as the cast compares them (``1`` and ``" 1 "`` are
                # level ``"1"``); raw values gave spurious warnings.
                present_values = {self._normalize_str(v) for v in df[col].dropna().unique()}
                expected_levels = {self._normalize_str(v) for v in col_spec.levels}
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
