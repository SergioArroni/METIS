"""Uniform noise generator for calibration.

Generates random uniform data simulating the worst-case scenario.
Conforms to the NoiseGenerator Protocol in metis.domain.contracts.
"""

import numpy as np
import pandas as pd

# Schema types that must always produce numeric noise regardless of
# the observed pandas dtype (e.g. TotalCharges stored as str in the CSV
# but declared ``continuous`` in the YAML schema).
_NUMERIC_SCHEMA_TYPES = frozenset({"continuous", "discrete"})


class UniformNoiseGenerator:
    """
    Uniform random noise generator.

    Generates completely random data with uniform distributions,
    respecting the schema (column types) of the reference dataset.

    Parameters
    ----------
    schema : dict[str, str] | None
        Mapping ``{column_name: semantic_type}`` from the YAML config.
        When provided the generator uses the declared type instead of
        relying on the pandas dtype, which avoids the mismatch where a
        column like *TotalCharges* is stored as ``str`` but defined as
        ``continuous``.
    """

    def __init__(self, schema: dict[str, str] | None = None) -> None:
        self._schema = schema or {}

    def generate(self, reference_data: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
        """
        Generate uniform noise based on reference schema.

        Per-column strategy:
        - Numeric (or schema-continuous/discrete): Uniform [min, max]
        - Categorical: Uniform sampling from unique values
        - Boolean: 50/50 True/False

        Args:
            reference_data: Reference dataset (for schema)
            n_samples: Number of rows to generate
            seed: Random seed

        Returns:
            DataFrame with uniform noise
        """
        rng = np.random.RandomState(seed)
        noise_data = {}

        for col in reference_data.columns:
            schema_type = self._schema.get(col)
            dtype = reference_data[col].dtype
            is_numeric = pd.api.types.is_numeric_dtype(dtype)
            is_schema_numeric = schema_type in _NUMERIC_SCHEMA_TYPES

            if is_numeric or is_schema_numeric:
                # Coerce to numeric first (handles str-encoded numerics)
                numeric_col = pd.to_numeric(reference_data[col], errors="coerce")
                col_min = numeric_col.min()
                col_max = numeric_col.max()

                if pd.isna(col_min) or pd.isna(col_max):
                    # Column is entirely non-numeric — fall back to
                    # categorical sampling.
                    noise_data[col] = self._categorical_noise(
                        reference_data[col],
                        rng,
                        n_samples,
                    )
                elif pd.api.types.is_integer_dtype(dtype) and not is_schema_numeric:
                    try:
                        noise_data[col] = rng.randint(
                            int(col_min),
                            int(col_max) + 1,
                            size=n_samples,
                            dtype=np.int64,
                        )
                    except (ValueError, OverflowError):
                        noise_data[col] = rng.uniform(
                            col_min,
                            col_max,
                            size=n_samples,
                        ).astype(np.int64)
                else:
                    noise_data[col] = rng.uniform(col_min, col_max, size=n_samples)

            elif pd.api.types.is_bool_dtype(dtype):
                noise_data[col] = rng.choice([True, False], size=n_samples)

            else:
                noise_data[col] = self._categorical_noise(
                    reference_data[col],
                    rng,
                    n_samples,
                )

        return pd.DataFrame(noise_data)

    # ------------------------------------------------------------------
    @staticmethod
    def _categorical_noise(
        series: pd.Series,
        rng: np.random.RandomState,
        n_samples: int,
    ) -> pd.Series | np.ndarray:
        """Sample uniformly from unique non-null values."""
        unique_vals = series.dropna().unique()
        if len(unique_vals) == 0:
            return pd.Series(["MISSING"] * n_samples, dtype=object)
        if len(unique_vals) == 1:
            return pd.Series([unique_vals[0]] * n_samples)
        return rng.choice(unique_vals, size=n_samples)
