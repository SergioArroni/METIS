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

Aggregation into the privacy family score is done by the generic family
aggregator from the taxonomy (``metis.domain.taxonomy.PRIVACY_METRICS``).

All metrics are normalized to [0, 1] where 1 = most private.
"""

# Dataset-based metrics - Attacks
from .dataset_based.attacks import (
    InferenceAttackMetric,
    MembershipInferenceMetric,
    RecordLinkageMetric,
)

# Dataset-based metrics - Empirical Similarity
from .dataset_based.empirical_similarity import DCRMetric, NNAAMetric

# Dataset-based metrics - Syntactic
from .dataset_based.syntactic import KAnonymityMetric, LDiversityMetric, TClosenessMetric

# Mechanism-based metrics
from .mechanism_based import DifferentialPrivacyMetric

# Base classes
from .privacy_base import (
    AttackMetric,
    BasePrivacyMetric,
    DatasetPrivacyMetric,
    EmpiricalSimilarityMetric,
    MechanismPrivacyMetric,
    SyntacticMetric,
)

__all__ = [
    # Base classes
    "BasePrivacyMetric",
    "DatasetPrivacyMetric",
    "MechanismPrivacyMetric",
    "AttackMetric",
    "SyntacticMetric",
    "EmpiricalSimilarityMetric",
    # Attacks
    "MembershipInferenceMetric",
    "InferenceAttackMetric",
    # Syntactic
    "KAnonymityMetric",
    "LDiversityMetric",
    "TClosenessMetric",
    "RecordLinkageMetric",
    # Empirical Similarity
    "DCRMetric",
    "NNAAMetric",
    # Mechanism-based
    "DifferentialPrivacyMetric",
]
