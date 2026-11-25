"""
Distances between the joint distributions of the two tables.

- MMD: Maximum Mean Discrepancy (kernel-based)
- Energy Distance: Multivariate statistical distance
"""

from .energy_distance import EnergyDistanceMetric
from .mmd import MMDMetric

__all__ = [
    "EnergyDistanceMetric",
    "MMDMetric",
]
