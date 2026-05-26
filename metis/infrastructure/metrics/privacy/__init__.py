"""
Privacy metrics for measuring synthetic data privacy preservation.

This module provides comprehensive privacy assessment:

Categories:
1. Dataset-based Privacy Metrics:
   - Attribute Inference: MIA, Inference Attack
   - Reidentification: k-Anonymity, l-Diversity, t-Closeness, Record Linkage
   - Empirical Similarity: DCR, NNAA

2. Mechanism-based Privacy Metrics:
   - Differential Privacy

Aggregation uses Stochastic Dominance:
- FSD for Levels 1-2 (metrics → subcategories)
- SSD for Level 3 (subcategories → categories → final)

All metrics are normalized to [0, 1] where 1 = most private.
"""

# Dataset-based metrics - Attribute Inference
from .dataset_based.attribute_inference import InferenceAttackMetric, MembershipInferenceMetric

# Dataset-based metrics - Empirical Similarity
from .dataset_based.empirical_similarity import DCRMetric, NNAAMetric

# Dataset-based metrics - Reidentification
from .dataset_based.reidentification import (
    KAnonymityMetric,
    LDiversityMetric,
    RecordLinkageMetric,
    TClosenessMetric,
)

# Mechanism-based metrics
from .mechanism_based import DifferentialPrivacyMetric

# Aggregator
from .privacy_aggregator import PrivacyAggregator, PrivacyResult

# Base classes
from .privacy_base import (
    AttributeInferenceMetric,
    BasePrivacyMetric,
    DatasetPrivacyMetric,
    EmpiricalSimilarityMetric,
    MechanismPrivacyMetric,
    ReidentificationMetric,
)

__all__ = [
    # Base classes
    "BasePrivacyMetric",
    "DatasetPrivacyMetric",
    "MechanismPrivacyMetric",
    "AttributeInferenceMetric",
    "ReidentificationMetric",
    "EmpiricalSimilarityMetric",
    # Attribute Inference
    "MembershipInferenceMetric",
    "InferenceAttackMetric",
    # Reidentification
    "KAnonymityMetric",
    "LDiversityMetric",
    "TClosenessMetric",
    "RecordLinkageMetric",
    # Empirical Similarity
    "DCRMetric",
    "NNAAMetric",
    # Mechanism-based
    "DifferentialPrivacyMetric",
    # Aggregator
    "PrivacyAggregator",
    "PrivacyResult",
]
