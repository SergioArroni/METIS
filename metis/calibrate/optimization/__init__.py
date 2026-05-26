"""
Optimización de agregadores.

Contiene la lógica de tuning de funciones de agregación para
encontrar la mejor combinación que represente los bounds teóricos.
"""

from .aggregator_tuner import AggregatorTuner

__all__ = ["AggregatorTuner"]
