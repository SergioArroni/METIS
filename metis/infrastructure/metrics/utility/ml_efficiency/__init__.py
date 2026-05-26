"""
ML Efficiency metrics for measuring synthetic data utility in machine learning tasks.

This module evaluates synthetic data quality through 5 training/testing strategies:
- TTR: Train on Real, Test on Real (baseline reference)
- TTS: Train on Synthetic, Test on Synthetic (synthetic self-consistency)
- TRTS: Train on Real, Test on Synthetic (distribution shift detection)
- TSTR: Train on Synthetic, Test on Real (practical utility)
- TTRS: Train on Real+Synthetic cocktail, Test on Real (augmentation utility)

Each strategy is compared against TTR baseline using absolute difference.
Final score = 1 - mean(|metric - TTR|) normalized to [0, 1].
"""

from .base import BaseStrategyMetric, MLEfficiencyResult, StrategyResult, TrainingStrategy
from .catboost_trainer import CatBoostTrainer, ClassificationTrainer, RegressionTrainer
from .classification_efficiency import ClassificationEfficiencyMetric
from .regression_efficiency import RegressionEfficiencyMetric
from .trts import TRTSMetric
from .tstr import TSTRMetric
from .ttr import TTRMetric
from .ttrs import TTRSMetric
from .tts import TTSMetric

__all__ = [
    # Base
    "BaseStrategyMetric",
    "MLEfficiencyResult",
    "StrategyResult",
    "TrainingStrategy",
    # Trainers
    "CatBoostTrainer",
    "ClassificationTrainer",
    "RegressionTrainer",
    # Individual strategies
    "TTRMetric",
    "TTSMetric",
    "TRTSMetric",
    "TSTRMetric",
    "TTRSMetric",
    # Aggregated metrics
    "ClassificationEfficiencyMetric",
    "RegressionEfficiencyMetric",
]
