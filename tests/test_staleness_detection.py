"""Tests for the fixed staleness detection logic (Phase 2.1)."""

import pandas as pd
import numpy as np
import pytest

from regime_ml.data.macro.alignment import add_staleness_indicators, align_to_calendar


def _make_df(series_code: str, dates: list, values: list) -> pd.DataFrame:
    """Build a minimal pre-alignment DataFrame."""
    return pd.DataFrame({
        "series_code": [series_code] * len(dates),
        "date": pd.to_datetime(dates),
        "value": values,
        "series_name": [series_code] * len(dates),
        "category": ["test"] * len(dates),
    })


# ---------------------------------------------------------------------------
# 2.1 — is_new_data is unconditionally True pre-alignment
# ---------------------------------------------------------------------------

def test_repeated_value_both_get_is_new_data_true():
    """When a series publishes the same value twice, both rows must be True."""
    df = _make_df("CFNAI", ["2020-01-01", "2020-02-01"], [0.00, 0.00])
    calendar = pd.bdate_range("2020-01-01", "2020-03-01")
    result = add_staleness_indicators(df, calendar)
    assert result["is_new_data"].all(), "All pre-alignment rows must be is_new_data=True"


def test_single_row_is_new_data_true():
    """A single-row series must be is_new_data=True."""
    df = _make_df("VIX", ["2020-01-02"], [15.0])
    calendar = pd.bdate_range("2020-01-01", "2020-01-10")
    result = add_staleness_indicators(df, calendar)
    assert result["is_new_data"].iloc[0] == True


def test_all_rows_unconditionally_true():
    """Random-value series: every pre-alignment row must always be True."""
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame({
        "series_code": ["VIX"] * n,
        "date": pd.bdate_range("2020-01-01", periods=n),
        "value": rng.standard_normal(n),
        "series_name": ["VIX"] * n,
        "category": ["stress"] * n,
    })
    calendar = pd.bdate_range("2020-01-01", periods=n)
    result = add_staleness_indicators(df, calendar)
    assert (result["is_new_data"] == True).all()


def test_increasing_then_flat_values_all_true():
    """No value pattern should produce is_new_data=False pre-alignment."""
    values = [1.0, 2.0, 2.0, 2.0, 3.0, 3.0]  # Mix of increases and repeats
    df = _make_df("INDPRO", [
        "2020-01-01", "2020-02-01", "2020-03-01",
        "2020-04-01", "2020-05-01", "2020-06-01"
    ], values)
    calendar = pd.bdate_range("2020-01-01", "2020-07-01")
    result = add_staleness_indicators(df, calendar)
    assert result["is_new_data"].all()


# ---------------------------------------------------------------------------
# Forward-filled rows get NaN after align_to_calendar
# ---------------------------------------------------------------------------

def test_forward_filled_rows_get_nan_is_new_data():
    """After align_to_calendar, interpolated dates must have is_new_data=NaN."""
    df = _make_df(
        "CFNAI",
        ["2020-01-02", "2020-02-03"],
        [0.5, 0.6]
    )
    calendar = pd.bdate_range("2020-01-02", "2020-02-03")
    df_stale = add_staleness_indicators(df, calendar)
    df_aligned = align_to_calendar(df_stale, calendar)

    cfnai = df_aligned[df_aligned["series_code"] == "CFNAI"]

    # Rows between the two real observation dates should have NaN is_new_data
    between = cfnai[
        (cfnai["date"] > pd.Timestamp("2020-01-02")) &
        (cfnai["date"] < pd.Timestamp("2020-02-03"))
    ]
    assert between["is_new_data"].isna().all(), (
        "Forward-filled rows between real observations must have NaN is_new_data"
    )


def test_real_observation_dates_have_true_is_new_data_after_alignment():
    """After align_to_calendar, original publication dates must have is_new_data=True."""
    df = _make_df("VIX", ["2020-01-02", "2020-01-05"], [14.0, 15.0])
    calendar = pd.bdate_range("2020-01-02", "2020-01-10")
    df_stale = add_staleness_indicators(df, calendar)
    df_aligned = align_to_calendar(df_stale, calendar)

    vix = df_aligned[df_aligned["series_code"] == "VIX"]
    pub_dates = pd.to_datetime(["2020-01-02", "2020-01-05"])
    pub_rows = vix[vix["date"].isin(pub_dates)]
    assert (pub_rows["is_new_data"] == True).all(), (
        "Original publication dates must have is_new_data=True after alignment"
    )


# ---------------------------------------------------------------------------
# native_freq and obs_number still computed correctly
# ---------------------------------------------------------------------------

def test_native_freq_monthly_inferred():
    """Monthly series must get native_freq='monthly'."""
    dates = pd.date_range("2020-01-01", periods=6, freq="MS").strftime("%Y-%m-%d").tolist()
    df = _make_df("CFNAI", dates, [0.1] * 6)
    calendar = pd.bdate_range("2020-01-01", "2020-07-01")
    result = add_staleness_indicators(df, calendar)
    assert (result["native_freq"] == "monthly").all()


def test_obs_number_increments():
    """obs_number must be 1, 2, 3... for each series."""
    df = _make_df("VIX", ["2020-01-02", "2020-01-03", "2020-01-06"], [10, 11, 12])
    calendar = pd.bdate_range("2020-01-01", "2020-01-10")
    result = add_staleness_indicators(df, calendar)
    assert list(result["obs_number"]) == [1, 2, 3]
