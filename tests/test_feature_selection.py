"""Tests for get_top_features() runtime validation (1.8a)."""

import pytest

from regime_ml.features.macro.selection import get_top_features


def test_no_validation_when_available_features_not_provided():
    """With no available_features arg, function returns names without validation."""
    result = get_top_features(n=5)
    assert len(result) == 5
    assert all(isinstance(f, str) for f in result)


def test_valid_features_pass_validation():
    """No error when all requested names are present in available_features."""
    top5 = get_top_features(n=5)
    # Pass the exact set back — should not raise.
    result = get_top_features(n=5, available_features=top5)
    assert result == top5


def test_missing_feature_raises_value_error():
    """ValueError is raised when a requested name is absent from available_features."""
    with pytest.raises(ValueError, match="not present in the computed feature set"):
        get_top_features(n=3, available_features=["FAKE_feature_1", "FAKE_feature_2"])


def test_superset_available_features_passes():
    """available_features may contain extra names beyond the top-N; that is fine."""
    top5 = get_top_features(n=5)
    extra = top5 + ["EXTRA_feature_1", "EXTRA_feature_2"]
    result = get_top_features(n=5, available_features=extra)
    assert result == top5
