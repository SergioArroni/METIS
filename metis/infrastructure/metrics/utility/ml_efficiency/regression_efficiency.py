"""
Regression Efficiency Metric.

``utility.ml_efficiency`` with the task forced to regression (MAE).  It
shares the protocol, the absolute score and the TTR cache of ``utility.ml_efficiency`` (see ``ml_efficiency_metric``).
"""

from metis.infrastructure.metrics.registry import register

from .forced_task import _ForcedTaskEfficiencyMetric


@register("utility.regression_efficiency")
class RegressionEfficiencyMetric(_ForcedTaskEfficiencyMetric):
    """Regression ML efficiency (MAE), CatBoost with early stopping."""

    name: str = "regression_efficiency"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "ml_efficiency", "regression", "mae"}
    metric_id = "utility.regression_efficiency"
    task_type = "regression"
