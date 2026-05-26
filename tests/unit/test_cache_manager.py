"""Tests for metis.calibrate.cache.cache_manager module."""

import pandas as pd
import pytest

from metis.calibrate.cache.cache_manager import CacheManager


@pytest.fixture
def cache_dir(tmp_path):
    """Temporary cache directory."""
    return tmp_path / "cache"


@pytest.fixture
def sample_df():
    """Small DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 45.0],
            "income": [30000.0, 50000.0, 70000.0],
        }
    )


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_init_creates_cache_dir(self, cache_dir):
        CacheManager(cache_dir=str(cache_dir))
        assert cache_dir.exists()

    def test_init_with_dataset_name(self, cache_dir):
        mgr = CacheManager(cache_dir=str(cache_dir), dataset_name="cardio")
        assert mgr.dataset_name == "cardio"

    def test_get_cache_path_includes_dataset_name(self, cache_dir):
        mgr = CacheManager(cache_dir=str(cache_dir), dataset_name="cardio")
        path = mgr._get_cache_path("test_key_123", date_tag="20260101")
        assert "cardio" in path.name
        assert "test_key_123" in path.name
        assert "20260101" in path.name

    def test_get_cache_path_without_dataset_name(self, cache_dir):
        mgr = CacheManager(cache_dir=str(cache_dir))
        path = mgr._get_cache_path("test_key_123", date_tag="20260101")
        assert path.name == "bounds_test_key_123_20260101.json"

    def test_find_cache_path_returns_none_when_empty(self, cache_dir):
        mgr = CacheManager(cache_dir=str(cache_dir))
        assert mgr._find_cache_path("nonexistent_key") is None

    def test_find_cache_path_finds_existing(self, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Create a fake cache file
        fake_file = cache_dir / "bounds_my_key_20260501.json"
        fake_file.write_text("{}")

        mgr = CacheManager(cache_dir=str(cache_dir))
        found = mgr._find_cache_path("my_key")
        assert found == fake_file

    def test_find_cache_path_returns_most_recent(self, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Create two cache files with different dates
        old = cache_dir / "bounds_my_key_20260101.json"
        new = cache_dir / "bounds_my_key_20260515.json"
        old.write_text("{}")
        new.write_text("{}")

        mgr = CacheManager(cache_dir=str(cache_dir))
        found = mgr._find_cache_path("my_key")
        assert found == new

    def test_invalidate_cache_specific_key(self, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        mgr = CacheManager(cache_dir=str(cache_dir))

        # Create a cache file matching how _get_cache_path generates it
        from datetime import datetime

        date_tag = datetime.now().strftime("%Y%m%d")
        path = cache_dir / f"bounds_key_to_delete_{date_tag}.json"
        path.write_text("{}")
        assert path.exists()

        mgr.invalidate_cache("key_to_delete")
        assert not path.exists()

    def test_invalidate_cache_all(self, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Create multiple cache files
        (cache_dir / "bounds_a_20260101.json").write_text("{}")
        (cache_dir / "bounds_b_20260102.json").write_text("{}")

        mgr = CacheManager(cache_dir=str(cache_dir))
        mgr.invalidate_cache()

        remaining = list(cache_dir.glob("*.json"))
        assert len(remaining) == 0

    def test_long_filename_gets_truncated(self, cache_dir):
        mgr = CacheManager(cache_dir=str(cache_dir), dataset_name="very_long_dataset_name")
        # Create a key that would make filename > 150 chars
        long_key = "a" * 200
        path = mgr._get_cache_path(long_key, date_tag="20260101")
        assert len(path.name) <= 150
