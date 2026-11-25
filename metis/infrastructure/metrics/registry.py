"""Metric registry with decorator-based registration."""

from collections.abc import Callable

from ...domain.contracts import Metric
from ...domain.errors import RegistryError


class MetricRegistryImpl:
    """Concrete implementation of metric registry."""

    def __init__(self):
        self._metrics: dict[str, type[Metric]] = {}
        self._defaults_loaded = False

    def _ensure_defaults(self) -> None:
        """Import the built-in metrics on first use, not when this module loads.

        Registering them at import time made every metric module circular:
        importing ``chi2_stat`` first imported the registry, which imported
        ``chi2_stat`` again while it was half-initialised and failed.
        """
        if not self._defaults_loaded:
            self._defaults_loaded = True
            _register_default_metrics()

    def register(self, metric_id: str, metric_class: type[Metric]) -> None:
        """Register a metric implementation."""
        if not metric_id:
            raise RegistryError("Metric ID cannot be empty", "metric")

        if metric_id in self._metrics:
            raise RegistryError(f"Metric already registered: {metric_id}", "metric", metric_id)

        self._metrics[metric_id] = metric_class

    def get(self, metric_id: str) -> type[Metric]:
        """Retrieve metric class by ID."""
        self._ensure_defaults()
        if metric_id not in self._metrics:
            available = list(self._metrics.keys())
            raise RegistryError(
                f"Metric not found: {metric_id}. Available: {available}",
                "metric",
                metric_id,
            )
        return self._metrics[metric_id]

    def list_ids(self, family: str | None = None) -> list[str]:
        """list available metric IDs, optionally filtered by family."""
        self._ensure_defaults()
        if family is None:
            return list(self._metrics.keys())

        # Filter by family prefix
        return [mid for mid in self._metrics if mid.startswith(f"{family}.")]

    def list_by_family(self) -> dict[str, list[str]]:
        """list metrics grouped by family."""
        self._ensure_defaults()
        families = {}
        for metric_id in self._metrics:
            if "." in metric_id:
                family = metric_id.split(".")[0]
                if family not in families:
                    families[family] = []
                families[family].append(metric_id)
        return families

    def list_by_hierarchy(self) -> dict[str, dict[str, dict[str, list[str]]]]:
        """
        list metrics grouped by full hierarchy (family -> category -> subcategory).

        Returns:
            Nested dictionary: {family: {category: {subcategory: [metric_ids]}}}
        """
        from ...domain.taxonomy import group_metrics_by_hierarchy

        self._ensure_defaults()
        return group_metrics_by_hierarchy(list(self._metrics.keys()))

    def get_metric_hierarchy(self, metric_id: str) -> dict[str, str]:
        """
        Get the hierarchy path for a registered metric.

        Args:
            metric_id: The metric ID

        Returns:
            Dictionary with family, category, subcategory
        """
        from ...domain.taxonomy import get_metric_hierarchy

        return get_metric_hierarchy(metric_id)


# Global registry instance
_metric_registry = MetricRegistryImpl()


def get_metric_registry() -> MetricRegistryImpl:
    """Get global metric registry instance."""
    return _metric_registry


def register(metric_id: str) -> Callable[[type[Metric]], type[Metric]]:
    """
    Decorator for registering metrics in the global registry.

    Usage:
        @register("fidelity.ks")
        class KSTestMetric:
            ...
    """

    def decorator(metric_class: type[Metric]) -> type[Metric]:
        _metric_registry.register(metric_id, metric_class)
        return metric_class

    return decorator


def register_metric(metric_id: str, metric_class: type[Metric]) -> None:
    """Register a metric in the global registry (alternative to decorator)."""
    _metric_registry.register(metric_id, metric_class)


# Import and register default metrics
def _register_default_metrics():
    """Register all built-in metrics."""
    # Import modules to trigger registration via decorators

    # Fidelity metrics - Conditional - Cat↔Cat
    from .fidelity.conditional.cat_cat.chi2_stat import Chi2StatMetric
    from .fidelity.conditional.cat_cat.cramers_v import CramersVMetric
    from .fidelity.conditional.cat_cat.theils_u import TheilsUMetric

    # Fidelity metrics - Conditional - Num↔Cat
    from .fidelity.conditional.num_cat.eta_squared import EtaSquaredMetric
    from .fidelity.conditional.num_cat.kruskal_epsilon import KruskalEpsilonMetric
    from .fidelity.conditional.num_cat.point_biserial import PointBiserialMetric

    # Fidelity metrics - Conditional - Num↔Num
    from .fidelity.conditional.num_num.correlation_matrix import CorrelationMatrixMetric
    from .fidelity.conditional.num_num.dcor import DistanceCorrelationMetric
    from .fidelity.conditional.num_num.mi import MutualInformationMetric
    from .fidelity.conditional.num_num.pearson import PearsonCorrelationMetric
    from .fidelity.conditional.num_num.spearman import SpearmanCorrelationMetric

    # Fidelity metrics - Global - Detection and Distances
    from .fidelity.global_metrics.detection.discriminator import DiscriminatorMetric
    from .fidelity.global_metrics.distances.energy_distance import EnergyDistanceMetric
    from .fidelity.global_metrics.distances.mmd import MMDMetric

    # Fidelity metrics - Marginal - Coverage
    from .fidelity.marginal.coverage.entropy_delta import ShannonEntropyDeltaMetric
    from .fidelity.marginal.coverage.gini_delta import GiniDeltaMetric
    from .fidelity.marginal.coverage.js import JSDivergenceMetric
    from .fidelity.marginal.coverage.kl import KLDivergenceMetric
    from .fidelity.marginal.coverage.psi import PSIMetric
    from .fidelity.marginal.coverage.tvd import TVDMetric

    # Fidelity metrics - Marginal - Scales
    from .fidelity.marginal.scales.cohens_d import CohensD
    from .fidelity.marginal.scales.delta_iqr import DeltaIQRMetric
    from .fidelity.marginal.scales.delta_mad import DeltaMADMetric
    from .fidelity.marginal.scales.delta_mean import DeltaMeanMetric
    from .fidelity.marginal.scales.delta_median import DeltaMedianMetric

    # Fidelity metrics - Marginal - Tails
    from .fidelity.marginal.tails.anderson_darling import AndersonDarlingMetric
    from .fidelity.marginal.tails.delta_exceedance import DeltaExceedanceMetric
    from .fidelity.marginal.tails.hellinger import HellingerMetric
    from .fidelity.marginal.tails.kde_ise import KDEISEMetric
    from .fidelity.marginal.tails.ks import KSMetric
    from .fidelity.marginal.tails.outliers_coverage import OutliersCoverageMetric
    from .fidelity.marginal.tails.wasserstein import WassersteinMetric

    # Privacy metrics
    from .privacy.dataset_based.attacks import inference_attack, mia, record_linkage
    from .privacy.dataset_based.empirical_similarity import dcr, nnaa
    from .privacy.dataset_based.syntactic import k_anonymity, l_diversity, t_closeness
    from .privacy.mechanism_based import differential_privacy

    # Utility metrics
    from .utility.ml_efficiency import classification_efficiency, regression_efficiency

    # Suppress unused import warnings - imports trigger decorator registration
    _ = (
        # Fidelity - Marginal - Tails
        KSMetric,
        WassersteinMetric,
        AndersonDarlingMetric,
        HellingerMetric,
        KDEISEMetric,
        DeltaExceedanceMetric,
        OutliersCoverageMetric,
        # Fidelity - Marginal - Scales
        DeltaMeanMetric,
        DeltaMedianMetric,
        DeltaIQRMetric,
        DeltaMADMetric,
        CohensD,
        # Fidelity - Marginal - Coverage
        TVDMetric,
        JSDivergenceMetric,
        KLDivergenceMetric,
        PSIMetric,
        ShannonEntropyDeltaMetric,
        GiniDeltaMetric,
        # Fidelity - Conditional - Num↔Num
        PearsonCorrelationMetric,
        SpearmanCorrelationMetric,
        DistanceCorrelationMetric,
        MutualInformationMetric,
        CorrelationMatrixMetric,
        # Fidelity - Conditional - Num↔Cat
        EtaSquaredMetric,
        KruskalEpsilonMetric,
        PointBiserialMetric,
        # Fidelity - Conditional - Cat↔Cat
        CramersVMetric,
        Chi2StatMetric,
        TheilsUMetric,
        # Fidelity - Global - Distances and Detection
        MMDMetric,
        EnergyDistanceMetric,
        DiscriminatorMetric,
        # Privacy
        mia,
        inference_attack,
        k_anonymity,
        l_diversity,
        t_closeness,
        record_linkage,
        dcr,
        nnaa,
        differential_privacy,
        # Utility
        classification_efficiency,
        regression_efficiency,
    )
