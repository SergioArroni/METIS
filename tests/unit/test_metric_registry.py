"""Tests for metis.infrastructure.metrics.registry module."""

import pytest

from metis.domain.errors import RegistryError
from metis.infrastructure.metrics.registry import MetricRegistryImpl, get_metric_registry


class TestMetricRegistryImpl:
    """Tests for MetricRegistryImpl class."""

    def test_register_and_get(self):
        reg = MetricRegistryImpl()

        class FakeMetric:
            pass

        reg.register("fidelity.test_metric", FakeMetric)
        assert reg.get("fidelity.test_metric") is FakeMetric

    def test_register_empty_id_raises(self):
        reg = MetricRegistryImpl()

        class FakeMetric:
            pass

        with pytest.raises(RegistryError):
            reg.register("", FakeMetric)

    def test_register_duplicate_raises(self):
        reg = MetricRegistryImpl()

        class FakeMetric:
            pass

        reg.register("fidelity.dup", FakeMetric)
        with pytest.raises(RegistryError, match="already registered"):
            reg.register("fidelity.dup", FakeMetric)

    def test_get_unknown_raises(self):
        reg = MetricRegistryImpl()
        with pytest.raises(RegistryError, match="not found"):
            reg.get("nonexistent.metric")

    def test_list_ids_all(self):
        reg = MetricRegistryImpl()

        class M1:
            pass

        class M2:
            pass

        reg.register("fidelity.a", M1)
        reg.register("privacy.b", M2)
        ids = reg.list_ids()
        assert "fidelity.a" in ids
        assert "privacy.b" in ids

    def test_list_ids_filtered_by_family(self):
        reg = MetricRegistryImpl()

        class M1:
            pass

        class M2:
            pass

        reg.register("fidelity.x", M1)
        reg.register("privacy.y", M2)
        assert reg.list_ids("fidelity") == ["fidelity.x"]
        assert reg.list_ids("privacy") == ["privacy.y"]

    def test_list_by_family(self):
        reg = MetricRegistryImpl()

        class M1:
            pass

        class M2:
            pass

        class M3:
            pass

        reg.register("fidelity.a", M1)
        reg.register("fidelity.b", M2)
        reg.register("utility.c", M3)

        families = reg.list_by_family()
        assert set(families["fidelity"]) == {"fidelity.a", "fidelity.b"}
        assert families["utility"] == ["utility.c"]


class TestGlobalRegistry:
    """Tests for the global registry singleton and decorator."""

    def test_global_registry_returns_instance(self):
        reg = get_metric_registry()
        assert isinstance(reg, MetricRegistryImpl)

    def test_global_registry_has_registered_metrics(self):
        reg = get_metric_registry()
        ids = reg.list_ids()
        # Should have the default metrics registered
        assert "fidelity.ks" in ids
        assert "fidelity.wasserstein" in ids
        assert "privacy.dcr" in ids

    def test_register_decorator(self):
        reg = get_metric_registry()
        # The global registry should already have metrics from _register_default_metrics
        metric_cls = reg.get("fidelity.ks")
        assert metric_cls is not None
