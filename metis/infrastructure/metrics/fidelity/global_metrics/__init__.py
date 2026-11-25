"""
Global fidelity metrics package: relations among more than two variables, the record as a whole.

Subcategories:
- distances: MMD and energy distance between the joint distributions
- detection: a classifier that tries to tell real rows from synthetic ones
"""

from .detection import DiscriminatorMetric
from .distances import EnergyDistanceMetric, MMDMetric

__all__ = [
    "DiscriminatorMetric",
    "EnergyDistanceMetric",
    "MMDMetric",
]
