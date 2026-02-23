"""Tests for validate_gaussian_assumption() in evaluation.py (item 3.3b)."""

import numpy as np
import pytest

from regime_ml.regimes.evaluation import validate_gaussian_assumption


def _make_X_and_labels(seed: int = 0, n: int = 200, d: int = 3, n_regimes: int = 3):
    """Synthetic balanced data with integer regime labels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    labels = np.array([i % n_regimes for i in range(n)])
    return X, labels


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_output_structure():
    """Output must be a dict keyed by int regime ids; each sub-dict has all 7 fields."""
    X, labels = _make_X_and_labels()
    result = validate_gaussian_assumption(X, labels, feature_names=["a", "b", "c"])
    assert set(result.keys()) == {0, 1, 2}
    required_fields = {"n", "skewness", "kurtosis", "shapiro_W", "shapiro_p",
                       "qq_theoretical", "qq_sample"}
    for regime_stats in result.values():
        assert set(regime_stats.keys()) == {"a", "b", "c"}
        for feat_stats in regime_stats.values():
            assert required_fields.issubset(feat_stats.keys())


def test_default_feature_names():
    """When feature_names=None, keys must default to 'f0', 'f1', etc."""
    X, labels = _make_X_and_labels()
    result = validate_gaussian_assumption(X, labels)
    for regime_stats in result.values():
        assert set(regime_stats.keys()) == {"f0", "f1", "f2"}


def test_n_field_equals_regime_size():
    """The 'n' field must equal the number of rows assigned to that regime."""
    X, labels = _make_X_and_labels()
    result = validate_gaussian_assumption(X, labels)
    for regime_id, regime_stats in result.items():
        expected_n = int(np.sum(labels == regime_id))
        for feat_stats in regime_stats.values():
            assert feat_stats["n"] == expected_n


def test_qq_arrays_have_matching_length():
    """qq_theoretical and qq_sample must have the same length."""
    X, labels = _make_X_and_labels()
    result = validate_gaussian_assumption(X, labels)
    for regime_stats in result.values():
        for feat_stats in regime_stats.values():
            assert len(feat_stats["qq_theoretical"]) == len(feat_stats["qq_sample"])


# ---------------------------------------------------------------------------
# Statistical correctness
# ---------------------------------------------------------------------------

def test_skewness_near_zero_for_normal_data():
    """Standard normal data should yield |skewness| < 1.0 for large n."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((500, 2))
    labels = np.zeros(500, dtype=int)
    result = validate_gaussian_assumption(X, labels)
    assert abs(result[0]["f0"]["skewness"]) < 1.0
    assert abs(result[0]["f1"]["skewness"]) < 1.0


def test_kurtosis_near_zero_for_normal_data():
    """Excess kurtosis should be near 0 for Gaussian data (large n)."""
    rng = np.random.default_rng(88)
    X = rng.standard_normal((600, 1))
    labels = np.zeros(600, dtype=int)
    result = validate_gaussian_assumption(X, labels)
    assert abs(result[0]["f0"]["kurtosis"]) < 1.5


def test_non_gaussian_data_low_shapiro_p():
    """Heavily non-Gaussian data (exponential) should produce low Shapiro-Wilk p-value."""
    rng = np.random.default_rng(42)
    X = rng.exponential(scale=1.0, size=(200, 1))
    labels = np.zeros(200, dtype=int)
    result = validate_gaussian_assumption(X, labels)
    assert result[0]["f0"]["shapiro_p"] < 0.05


def test_high_kurtosis_for_heavy_tailed_data():
    """Student-t with low df has heavy tails → excess kurtosis >> 0."""
    rng = np.random.default_rng(7)
    X = rng.standard_t(df=3, size=(300, 1))
    labels = np.zeros(300, dtype=int)
    result = validate_gaussian_assumption(X, labels)
    assert result[0]["f0"]["kurtosis"] > 2.0, (
        "Expected excess kurtosis > 2 for t(3) distribution."
    )


# ---------------------------------------------------------------------------
# Small-regime handling
# ---------------------------------------------------------------------------

def test_shapiro_nan_for_small_regime():
    """Regimes with fewer than 8 observations must have shapiro_W = NaN."""
    X = np.random.default_rng(1).standard_normal((20, 2))
    labels = np.zeros(20, dtype=int)
    labels[:5] = 1    # regime 1 has only 5 points — below the 8-observation threshold
    result = validate_gaussian_assumption(X, labels)
    assert np.isnan(result[1]["f0"]["shapiro_W"])
    assert np.isnan(result[1]["f0"]["shapiro_p"])


def test_shapiro_present_for_regime_at_threshold():
    """Regimes with exactly 8 observations must have a finite shapiro_W."""
    X = np.random.default_rng(2).standard_normal((50, 1))
    labels = np.zeros(50, dtype=int)
    labels[:8] = 1    # regime 1 has exactly 8 points
    result = validate_gaussian_assumption(X, labels)
    assert np.isfinite(result[1]["f0"]["shapiro_W"])
    assert np.isfinite(result[1]["f0"]["shapiro_p"])


# ---------------------------------------------------------------------------
# Input robustness
# ---------------------------------------------------------------------------

def test_nan_in_features_handled_gracefully():
    """NaN values in X must not cause an exception; statistics computed on valid rows."""
    X, labels = _make_X_and_labels(n=100)
    X[0, 0] = np.nan
    X[10, 1] = np.nan
    result = validate_gaussian_assumption(X, labels)
    # All regimes must still produce output without error.
    assert set(result.keys()) == {0, 1, 2}
