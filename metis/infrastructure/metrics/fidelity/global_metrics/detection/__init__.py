"""
Detection fidelity metrics package.

Can a classifier tell real rows from synthetic ones?
- Discriminator: CatBoost two-sample test on the whole rows
"""

from .discriminator import DiscriminatorMetric

__all__ = ["DiscriminatorMetric"]
