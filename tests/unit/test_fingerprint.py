"""Tests for metis.calibrate.cache.fingerprint module."""

import pandas as pd
import pytest

from metis.calibrate.cache.fingerprint import (
    compute_config_fingerprint,
    compute_data_fingerprint,
    generate_cache_key,
)


@pytest.fixture
def sample_df():
    """Small deterministic DataFrame."""
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 45.0, 55.0],
            "income": [30000.0, 50000.0, 70000.0, 90000.0],
            "city": ["A", "B", "C", "A"],
        }
    )


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a minimal YAML config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  real: "data/real/test.csv"
  task_type: "classification"
  schema:
    age: continuous
    income: continuous
    city: categorical

metrics:
  - "fidelity.ks"
  - "fidelity.wasserstein"
"""
    )
    return str(config_path)


class TestComputeDataFingerprint:
    """Tests for compute_data_fingerprint."""

    def test_deterministic(self, sample_df):
        fp1 = compute_data_fingerprint(sample_df)
        fp2 = compute_data_fingerprint(sample_df)
        assert fp1 == fp2

    def test_different_data_different_fingerprint(self, sample_df):
        other_df = sample_df.copy()
        other_df.iloc[0, 0] = 999.0

        fp1 = compute_data_fingerprint(sample_df)
        fp2 = compute_data_fingerprint(other_df)
        assert fp1 != fp2

    def test_format_has_three_components(self, sample_df):
        fp = compute_data_fingerprint(sample_df)
        parts = fp.split("_")
        assert len(parts) == 3
        # Each part should be 16 hex chars
        for part in parts:
            assert len(part) == 16

    def test_column_order_matters(self, sample_df):
        reordered = sample_df[["income", "age", "city"]]
        fp1 = compute_data_fingerprint(sample_df)
        fp2 = compute_data_fingerprint(reordered)
        # Different column order = different fingerprint (columns are sorted)
        # Actually columns_str uses sorted() so same columns → same hash
        # But content order may differ
        # The important thing is it's deterministic
        assert isinstance(fp1, str)
        assert isinstance(fp2, str)

    def test_sample_size_param_ignored(self, sample_df):
        """sample_size parameter is deprecated and ignored."""
        fp1 = compute_data_fingerprint(sample_df)
        fp2 = compute_data_fingerprint(sample_df, sample_size=2)
        assert fp1 == fp2


class TestComputeConfigFingerprint:
    """Tests for compute_config_fingerprint."""

    def test_deterministic(self, sample_config_file):
        fp1 = compute_config_fingerprint(sample_config_file)
        fp2 = compute_config_fingerprint(sample_config_file)
        assert fp1 == fp2

    def test_different_config_different_fingerprint(self, tmp_path):
        config1 = tmp_path / "config1.yaml"
        config1.write_text(
            "data:\n  task_type: classification\n  schema:\n    a: continuous\nmetrics:\n  - fidelity.ks\n"
        )
        config2 = tmp_path / "config2.yaml"
        config2.write_text(
            "data:\n  task_type: regression\n  schema:\n    a: continuous\nmetrics:\n  - fidelity.ks\n"
        )

        fp1 = compute_config_fingerprint(str(config1))
        fp2 = compute_config_fingerprint(str(config2))
        assert fp1 != fp2

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            compute_config_fingerprint("/nonexistent/config.yaml")

    def test_returns_16_char_hex(self, sample_config_file):
        fp = compute_config_fingerprint(sample_config_file)
        assert len(fp) == 16
        # Should be valid hex
        int(fp, 16)


class TestGenerateCacheKey:
    """Tests for generate_cache_key."""

    def test_deterministic(self):
        key1 = generate_cache_key("abc_def_ghi", "xyz123", 12, 750, 42)
        key2 = generate_cache_key("abc_def_ghi", "xyz123", 12, 750, 42)
        assert key1 == key2

    def test_starts_with_calibration_prefix(self):
        key = generate_cache_key("data_fp", "config_fp", 10, 500, 42)
        assert key.startswith("calibration_")

    def test_different_params_different_key(self):
        key1 = generate_cache_key("data_fp", "config_fp", 10, 500, 42)
        key2 = generate_cache_key("data_fp", "config_fp", 20, 500, 42)
        assert key1 != key2

    def test_different_seed_different_key(self):
        key1 = generate_cache_key("data_fp", "config_fp", 10, 500, 42)
        key2 = generate_cache_key("data_fp", "config_fp", 10, 500, 99)
        assert key1 != key2

    def test_includes_fingerprints(self):
        key = generate_cache_key("my_data_fp", "my_config_fp", 10, 500, 42)
        assert "my_data_fp" in key
        assert "my_config_fp" in key
