"""
Utilidades para calibración.

Agrupa componentes reutilizables:
- Evaluador en memoria (optimizado)
- Generador de ruido uniforme
"""

from .evaluator import InMemoryEvaluator
from .noise_generator import UniformNoiseGenerator

__all__ = ["InMemoryEvaluator", "UniformNoiseGenerator"]
