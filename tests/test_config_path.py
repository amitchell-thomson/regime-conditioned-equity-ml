"""Tests for MACRO_DATA_PATH env var override and loaders.py default path removal."""

import pytest

from regime_ml.data.macro.loaders import load_raw_data


def test_load_raw_data_no_args_raises():
    """load_raw_data() with no source_dir must raise ValueError with a helpful message."""
    with pytest.raises(ValueError, match="source_dir is required"):
        load_raw_data()


def test_load_raw_data_none_raises():
    """load_raw_data(source_dir=None) must raise ValueError."""
    with pytest.raises(ValueError, match="source_dir is required"):
        load_raw_data(source_dir=None)


def test_macro_data_path_env_var_overrides_config(monkeypatch, tmp_path):
    """MACRO_DATA_PATH environment variable must override the YAML data_path."""
    monkeypatch.setenv("MACRO_DATA_PATH", str(tmp_path))

    from regime_ml.utils.config import load_configs

    cfg = load_configs()
    resolved_path = cfg["macro_data"]["regime_universe"]["data_path"]
    assert resolved_path == str(tmp_path), f"Expected MACRO_DATA_PATH={tmp_path} to override YAML, got {resolved_path}"


def test_macro_data_path_not_set_uses_yaml(monkeypatch):
    """When MACRO_DATA_PATH is unset, config should return the YAML value."""
    monkeypatch.delenv("MACRO_DATA_PATH", raising=False)

    from regime_ml.utils.config import load_configs

    cfg = load_configs()
    path = cfg["macro_data"]["regime_universe"]["data_path"]
    # The YAML value is a non-empty string (we don't validate it exists on disk)
    assert isinstance(path, str) and len(path) > 0
