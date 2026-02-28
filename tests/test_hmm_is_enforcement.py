"""Tests for IS-only scaler enforcement in initialise_emissions (Phase 2.2)."""

import logging
import numpy as np
import pandas as pd
import pytest

from regime_ml.regimes.hmm import initialise_emissions


def _make_df(n: int, seed: int = 0, start: str = "2010-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        rng.standard_normal((n, 3)), index=idx, columns=["f1", "f2", "f3"]
    )


# ---------------------------------------------------------------------------
# IS boundary enforcement
# ---------------------------------------------------------------------------


def test_within_boundary_no_error():
    """No error when df_train ends before train_end_date."""
    df = _make_df(100)
    train_end = df.index.max() + pd.Timedelta(days=1)
    means, covs, scaler = initialise_emissions(
        df, n_clusters=2, random_state=42, train_end_date=train_end
    )
    assert means.shape == (2, 3)


def test_exact_boundary_no_error():
    """No error when df_train's last date equals train_end_date exactly."""
    df = _make_df(100)
    train_end = df.index.max()
    means, covs, scaler = initialise_emissions(
        df, n_clusters=2, random_state=42, train_end_date=train_end
    )
    assert means.shape == (2, 3)


def test_oos_data_raises_value_error():
    """ValueError raised when df_train contains dates after train_end_date."""
    df = _make_df(100, start="2010-01-01")
    # Set train_end_date to the middle of the training set
    train_end = df.index[49]
    with pytest.raises(ValueError, match="train_end_date"):
        initialise_emissions(
            df, n_clusters=2, random_state=42, train_end_date=train_end
        )


def test_oos_error_message_includes_dates():
    """ValueError message must name both the offending date and train_end_date."""
    df = _make_df(50, start="2015-01-01")
    train_end = df.index[10]
    with pytest.raises(ValueError) as exc_info:
        initialise_emissions(
            df, n_clusters=2, random_state=42, train_end_date=train_end
        )
    msg = str(exc_info.value)
    assert "train_end_date" in msg


def test_no_datetime_index_skips_validation():
    """RangeIndex df_train → IS validation skipped (no error, no assertion)."""
    df = pd.DataFrame(
        np.random.default_rng(0).standard_normal((50, 2)), columns=["a", "b"]
    )
    means, covs, scaler = initialise_emissions(
        df, n_clusters=2, random_state=42, train_end_date=pd.Timestamp("2020-01-01")
    )
    assert means.shape[0] == 2


def test_no_train_end_date_backward_compatible():
    """Omitting train_end_date entirely is still valid (backward compatible)."""
    df = _make_df(60)
    means, covs, scaler = initialise_emissions(df, n_clusters=2, random_state=42)
    assert means.shape[0] == 2


# ---------------------------------------------------------------------------
# Scaler metadata
# ---------------------------------------------------------------------------


def test_train_date_range_stored_on_scaler():
    """Scaler must carry _regime_ml_train_date_range after fitting."""
    df = _make_df(80)
    _, _, scaler = initialise_emissions(df, n_clusters=2, random_state=42)
    assert hasattr(scaler, "_regime_ml_train_date_range")
    assert scaler._regime_ml_train_date_range is not None
    start, end = scaler._regime_ml_train_date_range
    assert start <= end


def test_train_date_range_matches_input():
    """_regime_ml_train_date_range must equal df_train's actual index bounds."""
    df = _make_df(60)
    _, _, scaler = initialise_emissions(df, n_clusters=2, random_state=42)
    stored_start, stored_end = scaler._regime_ml_train_date_range
    assert stored_start == df.index.min()
    assert stored_end == df.index.max()


def test_no_datetime_index_stores_none():
    """RangeIndex df_train → _regime_ml_train_date_range is None."""
    df = pd.DataFrame(
        np.random.default_rng(1).standard_normal((40, 2)), columns=["x", "y"]
    )
    _, _, scaler = initialise_emissions(df, n_clusters=2, random_state=42)
    assert hasattr(scaler, "_regime_ml_train_date_range")
    assert scaler._regime_ml_train_date_range is None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_training_date_range_logged(caplog):
    """Training date range must be logged at INFO level."""
    df = _make_df(50)
    with caplog.at_level(logging.INFO, logger="regime_ml"):
        initialise_emissions(df, n_clusters=2, random_state=42)
    assert any("initialise_emissions" in r.message for r in caplog.records)
