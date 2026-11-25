"""SDV 2 for a single table: metadata, fit and sample.

SDV 2 removed ``SingleTableMetadata`` and its synthesizers take and return a
dictionary of tables, also for one table.  The wrappers keep a single
``DataFrame`` interface and name the table here.
"""

from collections.abc import Iterable

import pandas as pd
from sdv.metadata import Metadata

TABLE = "table"


def metadata_for(
    data: pd.DataFrame, categorical: Iterable[str] = (), numerical: Iterable[str] = ()
) -> Metadata:
    """Detected metadata of *data* with the declared column types on top."""
    metadata = Metadata.detect_from_dataframes({TABLE: data})
    for sdtype, cols in (("categorical", categorical), ("numerical", numerical)):
        for col in cols:
            if col in data.columns:
                metadata.update_column(col, table_name=TABLE, sdtype=sdtype)
    return metadata


def fit(synthesizer, data: pd.DataFrame) -> None:
    synthesizer.fit({TABLE: data})


def sample(synthesizer, n_rows: int) -> pd.DataFrame:
    out = synthesizer.sample(num_rows=n_rows)
    return out[TABLE] if isinstance(out, dict) else out
