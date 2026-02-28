"""Tests for Winsorize transform (item 3.3a)."""

import numpy as np
import pandas as pd
import pytest

from regime_ml.features.common.transforms import TransformRegistry, Winsorize


def _make_series(n: int = 100, seed: int = 0) -> pd.Series:
    """Synthetic z-scored series with large variance to exercise the clip bounds."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.standard_normal(n) * 5, index=idx)


# ---------------------------------------------------------------------------
# Clipping behaviour
# ---------------------------------------------------------------------------


def test_clips_above_sigma():
    """Values above +sigma must be clipped to exactly +sigma."""
    s = pd.Series([0.0, 1.0, 5.0, 100.0])
    result = Winsorize(sigma=4.0)._compute(s)
    assert result.max() == pytest.approx(4.0)


def test_clips_below_minus_sigma():
    """Values below -sigma must be clipped to exactly -sigma."""
    s = pd.Series([-100.0, -5.0, 0.0, 1.0])
    result = Winsorize(sigma=4.0)._compute(s)
    assert result.min() == pytest.approx(-4.0)


def test_passes_values_within_sigma():
    """Values already within [-sigma, +sigma] must be unchanged."""
    s = pd.Series([-3.0, 0.0, 2.5])
    result = Winsorize(sigma=4.0)._compute(s)
    pd.testing.assert_series_equal(result, s, check_names=False)


def test_exact_boundary_values_unchanged():
    """Values at exactly ±sigma must be returned as-is (not changed)."""
    s = pd.Series([-4.0, 4.0])
    result = Winsorize(sigma=4.0)._compute(s)
    pd.testing.assert_series_equal(result, s, check_names=False)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_missing_sigma_raises():
    """Winsorize() without sigma must raise ValueError mentioning 'sigma'."""
    with pytest.raises(ValueError, match="sigma"):
        Winsorize()


def test_zero_sigma_raises():
    """Winsorize with sigma=0 must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        Winsorize(sigma=0.0)


def test_negative_sigma_raises():
    """Winsorize with sigma<0 must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        Winsorize(sigma=-1.0)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_passthrough():
    """NaN values must remain NaN after winsorization — never clipped to ±sigma."""
    s = pd.Series([np.nan, 5.0, -5.0, 0.0])
    result = Winsorize(sigma=4.0)._compute(s)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(4.0)
    assert result.iloc[2] == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered_in_registry():
    """'winsorize' key in TransformRegistry must resolve to Winsorize class."""
    assert TransformRegistry.get("winsorize") is Winsorize


def test_create_via_registry():
    """TransformRegistry.create('winsorize', sigma=4.0) must produce correct output."""
    w = TransformRegistry.create("winsorize", sigma=4.0)
    s = pd.Series([10.0, -10.0, 0.0])
    result = w._compute(s)
    assert result.max() == pytest.approx(4.0)
    assert result.min() == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# Staleness modes
# ---------------------------------------------------------------------------


def test_strict_staleness_mode_preserves_length():
    """Output length must equal input length under 'strict' staleness mode."""
    s = _make_series(n=50)
    is_new = pd.Series(False, index=s.index)
    is_new.iloc[::5] = True
    result = Winsorize(sigma=4.0).transform(
        s, is_new_data=is_new, staleness_mode="strict"
    )
    assert len(result) == len(s)


def test_strict_staleness_mode_clips_values():
    """Under 'strict' mode, non-NaN output values must all lie within [-sigma, +sigma]."""
    s = _make_series(n=60)
    is_new = pd.Series(True, index=s.index)
    result = Winsorize(sigma=4.0).transform(
        s, is_new_data=is_new, staleness_mode="strict"
    )
    valid = result.dropna()
    assert (valid.abs() <= 4.0 + 1e-9).all(), (
        f"Values outside ±4σ found: {valid[valid.abs() > 4.0 + 1e-9]}"
    )


def test_ignore_staleness_mode_clips_values():
    """Under 'ignore' mode, all output values must lie within [-sigma, +sigma]."""
    s = _make_series(n=50)
    result = Winsorize(sigma=4.0).transform(s, staleness_mode="ignore")
    assert (result.abs() <= 4.0 + 1e-9).all()
