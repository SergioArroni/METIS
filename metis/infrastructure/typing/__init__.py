"""Static column-type inference (ported from Hefesto's Phase -1).

>>> from metis.infrastructure.typing import infer_schema
>>> inferred = infer_schema(df)  # doctest: +SKIP
>>> inferred.schema["age"]  # doctest: +SKIP
{'type': 'discrete', 'step': 1.0, 'levels': None}
"""

from metis.infrastructure.typing.classification import Verdict, classify
from metis.infrastructure.typing.domain import Domain, measure
from metis.infrastructure.typing.inference import InferredSchema, infer_column, infer_schema
from metis.infrastructure.typing.thresholds import InferenceThresholds

__all__ = [
    "Domain",
    "InferenceThresholds",
    "InferredSchema",
    "Verdict",
    "classify",
    "infer_column",
    "infer_schema",
    "measure",
]
