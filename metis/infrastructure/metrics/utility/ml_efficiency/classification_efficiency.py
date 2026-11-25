"""
Classification Efficiency Metric.

``utility.ml_efficiency`` with the task forced to classification (weighted
F1).  It shares the protocol, the absolute score and the TTR cache of
``utility.ml_efficiency`` (see ``ml_efficiency_metric``); only the target
checks differ.
"""

from metis.infrastructure.metrics.registry import register

from .forced_task import _ForcedTaskEfficiencyMetric


@register("utility.classification_efficiency")
class ClassificationEfficiencyMetric(_ForcedTaskEfficiencyMetric):
    """Classification ML efficiency (weighted F1), CatBoost with early stopping."""

    name: str = "classification_efficiency"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "classification", "f1"}
    metric_id = "utility.classification_efficiency"
    task_type = "classification"
