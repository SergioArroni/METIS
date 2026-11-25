"""
Tails metrics package - Distribution tail comparison metrics.

Available metrics:
- KS: Kolmogorov-Smirnov test
- Wasserstein: Earth Mover's distance
- Hellinger: Hellinger distance
- Anderson-Darling: AD k-sample test
- Delta Exceedance: Exceedance probability difference
- KDE-ISE: KDE Integrated Squared Error
- Outliers coverage: ranges, outliers and density per column
"""

from metis.shared.results import ColumnMetricResult

from ...fidelity_base import NumericColumnMetric as TailMetric
from .anderson_darling import AndersonDarlingMetric
from .delta_exceedance import DeltaExceedanceMetric
from .hellinger import HellingerMetric
from .kde_ise import KDEISEMetric
from .ks import KSMetric
from .outliers_coverage import OutliersCoverageMetric
from .wasserstein import WassersteinMetric

__all__ = [
    "TailMetric",
    "ColumnMetricResult",
    "KSMetric",
    "WassersteinMetric",
    "HellingerMetric",
    "AndersonDarlingMetric",
    "DeltaExceedanceMetric",
    "KDEISEMetric",
    "OutliersCoverageMetric",
]
