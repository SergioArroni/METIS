"""Thresholds of the static type inference (Hefesto Phase -1).

Defaults are those of Hefesto's ``configs/metodo/inferencia_tipos.yaml`` (identical to the
module constants of ``hefesto/core/typing/umbrales.py``). Measuring (``domain``) and deciding
(``classification``) depend on this module without depending on each other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

# Hefesto's YAML keys -> field names here, so a Hefesto threshold file can be reused as is.
_HEFESTO_KEYS = {
    "umbral_degenerada": "degenerate",
    "umbral_cardinalidad_codigo": "code_cardinality",
    "umbral_palabras_texto_libre": "free_text_words",
    "umbral_longitud_texto_libre": "free_text_length",
    "umbral_ratio_unicidad_texto_libre": "free_text_uniqueness_ratio",
    "umbral_enumerable": "enumerable",
    "umbral_continuizacion": "continuization",
    "umbral_reticula": "lattice",
    "umbral_sin_reticula": "no_lattice",
}


@dataclass(frozen=True)
class InferenceThresholds:
    """Everything the inference compares.

    - ``degenerate``: a column with ``nunique <= degenerate`` is constant.
    - ``code_cardinality``: an integer column with at most this many distinct values is treated
      as a category code (conservative; only the user can say it is a quantity).
    - ``free_text_*``: free text if mean words/value > ``free_text_words``, OR mean chars/value >
      ``free_text_length`` AND distinct/rows > ``free_text_uniqueness_ratio``.
    - ``enumerable``: a numeric domain with at most this many distinct values is listed whole.
    - ``lattice``: fraction of contiguous gaps that must equal the modal gap for the lattice to
      be real (also applied on the log scale). NOT calibrated: measured on 61 columns of six
      real tables the margin is 0.006, and regularity grows with n even for continuous domains.
    - ``no_lattice``: below this the lattice is not visible; between ``no_lattice`` and
      ``lattice`` lies the undecided band.
    - ``continuization``: how many values fit in the p1-p99 core (at the modal step or the
      common grid). At or above, the grid is so fine the variable is continuised.
    """

    degenerate: int = 1
    code_cardinality: int = 10
    free_text_words: int = 20
    free_text_length: int = 150
    free_text_uniqueness_ratio: float = 0.3
    enumerable: int = 20
    continuization: int = 1000
    lattice: float = 0.8
    no_lattice: float = 0.4

    def __post_init__(self) -> None:
        if not self.no_lattice < self.lattice:
            raise ValueError(
                f"no_lattice ({self.no_lattice}) must be lower than lattice ({self.lattice}): "
                "the undecided band lies between them"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> InferenceThresholds:
        """Build from a mapping with either these field names or Hefesto's YAML keys. Missing
        keys keep their default; unknown keys are ignored."""
        types = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            name = _HEFESTO_KEYS.get(key, key)
            if name in types:
                kwargs[name] = float(value) if types[name] == "float" else int(value)
        return cls(**kwargs)
