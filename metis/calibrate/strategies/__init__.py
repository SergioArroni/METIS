"""
Estrategias de calibración (Strategy Pattern).

- UpperBoundStrategy: Real vs Real (split-half)
- LowerBoundStrategy: Real vs Noise (uniform random)
- BaseCalibrationStrategy: Lógica común (paralelización, merge, logging)
"""

from .lower_bound import LowerBoundStrategy
from .upper_bound import UpperBoundStrategy

__all__ = ["UpperBoundStrategy", "LowerBoundStrategy"]
