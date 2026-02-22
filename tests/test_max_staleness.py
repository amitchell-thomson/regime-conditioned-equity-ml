"""Tests for max_staleness_days enforcement in align_to_calendar (Phase 2.3)."""

import numpy as np
import pandas as pd
import pytest
import yaml

from regime_ml.data.macro.alignment import add_staleness_indicators, align_to_calendar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monthly_series(
    start: str = "2020-01-01",
    periods: int = 6,
    series_code: str = "TEST",
    value: float = 1.0,
) -> pd.DataFrame:
    """Monthly publication dates spaced ~30 days apart."""
    dates = pd.bdate_range(start, periods=periods, freq="BMS")
    return pd.DataFrame({
        "series_code": series_code,
        "date": dates,
        "value": value,
        "series_name": "Test Series",
        "category": "test",
    })


def _make_daily_series(
    start: str = "2020-01-01",
    periods: int = 30,
    series_code: str = "DAILY",
    value: float = 2.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({
        "series_code": series_code,
        "date": dates,
        "value": value,
        "series_name": "Daily Series",
        "category": "test",
    })


def _calendar(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_no_max_staleness_preserves_full_ffill():
    """max_staleness_days=None (default) → full forward-fill, no NaN-out."""
    # One monthly observation followed by 90 days of calendar
    df = _make_monthly_series(start="2020-01-02", periods=1)
    cal = _calendar("2020-01-01", "2020-05-01")  # ~90 business days

    with_staleness = add_staleness_indicators(df, cal)
    aligned = align_to_calendar(with_staleness, cal, max_staleness_days=None)

    aligned = aligned[aligned["series_code"] == "TEST"]
    # Every non-NaN value should be 1.0 (no NaN-out)
    non_null = aligned["value"].dropna()
    assert (non_null == 1.0).all(), "Full ffill should preserve all values without limit"


# ---------------------------------------------------------------------------
# Monthly series — 90-day gap with threshold=65
# ---------------------------------------------------------------------------

def test_monthly_series_nans_out_rows_beyond_threshold():
    """Monthly series: rows with days_since_update > 65 must be NaN."""
    # Single observation on Jan 2, 2020; calendar runs ~90 business days beyond it
    df = _make_monthly_series(start="2020-01-02", periods=1)
    cal = _calendar("2020-01-01", "2020-05-01")

    with_staleness = add_staleness_indicators(df, cal)
    aligned = align_to_calendar(
        with_staleness, cal, max_staleness_days={"monthly": 65, "unknown": 65}
    )
    aligned = aligned[aligned["series_code"] == "TEST"].dropna(subset=["date"])

    # Rows where days_since_update <= 65 must have value=1.0
    within = aligned[aligned["days_since_update"] <= 65]
    assert (within["value"].dropna() == 1.0).all()

    # Rows where days_since_update > 65 must have value=NaN
    beyond = aligned[aligned["days_since_update"] > 65]
    assert beyond["value"].isna().all(), (
        "Rows with days_since_update > 65 must be NaN for monthly series"
    )


def test_monthly_threshold_boundary_is_inclusive():
    """Row at exactly threshold days is kept (not NaN-out); threshold+1 is NaN."""
    df = _make_monthly_series(start="2020-01-02", periods=1)
    cal = _calendar("2020-01-01", "2020-06-01")

    with_staleness = add_staleness_indicators(df, cal)
    aligned = align_to_calendar(
        with_staleness, cal, max_staleness_days={"monthly": 65, "unknown": 65}
    )
    aligned = aligned[aligned["series_code"] == "TEST"]

    exactly_at = aligned[aligned["days_since_update"] == 65]
    if not exactly_at.empty:
        assert (exactly_at["value"] == 1.0).all(), "Row at threshold should NOT be NaN-out"

    just_beyond = aligned[aligned["days_since_update"] == 66]
    if not just_beyond.empty:
        assert just_beyond["value"].isna().all(), "Row at threshold+1 should be NaN"


# ---------------------------------------------------------------------------
# Daily series — 18-day gap with threshold=5
# ---------------------------------------------------------------------------

def test_daily_series_nans_out_beyond_threshold():
    """Daily series with a gap: rows with days_since_update > 5 must be NaN.

    Must have enough daily observations so infer_freq() returns 'daily'
    (median gap ≤ 1 day). We give 10 daily obs, then a 18-bday gap, then
    10 more.  The gap period should be NaN-out beyond threshold=5.
    """
    first_block = pd.bdate_range("2020-01-02", periods=10)     # daily obs
    second_block = pd.bdate_range("2020-02-10", periods=10)    # after ~27 bday gap
    dates = first_block.append(second_block)
    values = [10.0] * 10 + [20.0] * 10
    df = pd.DataFrame({
        "series_code": "DAILY",
        "date": dates,
        "value": values,
        "series_name": "Daily",
        "category": "test",
    })
    cal = _calendar("2020-01-01", "2020-03-01")

    with_staleness = add_staleness_indicators(df, cal)
    aligned = align_to_calendar(
        with_staleness, cal, max_staleness_days={"daily": 5, "monthly": 65, "unknown": 65}
    )
    aligned = aligned[aligned["series_code"] == "DAILY"]

    beyond = aligned[aligned["days_since_update"] > 5]
    assert beyond["value"].isna().all(), (
        "Daily series: rows with days_since_update > 5 must be NaN"
    )

    within = aligned[aligned["days_since_update"] <= 5]
    assert within["value"].dropna().notna().all()


def test_daily_series_second_observation_resets_staleness():
    """After a new daily observation, days_since_update resets to 0.

    Uses 10 daily observations so infer_freq() returns 'daily'.
    Checks that the last observation date has days_since_update=0.
    """
    # 2020-01-06 is the first Monday of the second week
    first_block = pd.bdate_range("2020-01-02", periods=9)   # Thu … next Tue
    second_date = pd.bdate_range("2020-01-20", periods=1)   # following Monday
    dates = first_block.append(second_date)
    values = [10.0] * 9 + [20.0]
    df = pd.DataFrame({
        "series_code": "DAILY",
        "date": dates,
        "value": values,
        "series_name": "Daily",
        "category": "test",
    })
    cal = _calendar("2020-01-01", "2020-01-31")

    with_staleness = add_staleness_indicators(df, cal)
    aligned = align_to_calendar(
        with_staleness, cal, max_staleness_days={"daily": 5, "monthly": 65, "unknown": 65}
    )
    aligned = aligned[aligned["series_code"] == "DAILY"]

    # Row on 2020-01-20 (the second publication) should have days_since_update=0
    on_update = aligned[aligned["date"] == pd.Timestamp("2020-01-20")]
    assert not on_update.empty, "2020-01-20 must be in the aligned calendar"
    assert on_update.iloc[0]["days_since_update"] == 0
    assert on_update.iloc[0]["value"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# YAML config contains required frequency keys
# ---------------------------------------------------------------------------

def test_yaml_config_has_required_staleness_keys():
    """configs/data/regime_universe.yaml must contain daily, weekly, monthly keys."""
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "configs" / "data" / "regime_universe.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    staleness = cfg.get("regime_universe", {}).get("max_staleness_days", {})
    for key in ("daily", "weekly", "monthly"):
        assert key in staleness, f"max_staleness_days missing key: {key}"
    # Values should be positive integers
    for key, val in staleness.items():
        assert isinstance(val, int) and val > 0, f"{key}: expected positive int, got {val!r}"


# ---------------------------------------------------------------------------
# Unknown / missing frequency falls back gracefully
# ---------------------------------------------------------------------------

def test_unknown_freq_uses_unknown_threshold():
    """Series with inferred 'irregular' or 'unknown' freq uses 'unknown' threshold."""
    # Two observations 45 days apart → might be 'irregular' depending on infer_freq
    dates = pd.DatetimeIndex(["2020-01-02", "2020-03-02"])
    df = pd.DataFrame({
        "series_code": "IRREG",
        "date": dates,
        "value": [5.0, 6.0],
        "series_name": "Irregular",
        "category": "test",
    })
    cal = _calendar("2020-01-01", "2020-05-01")

    with_staleness = add_staleness_indicators(df, cal)
    aligned = align_to_calendar(
        with_staleness, cal,
        max_staleness_days={"daily": 5, "weekly": 14, "monthly": 65, "irregular": 65, "unknown": 65}
    )
    aligned = aligned[aligned["series_code"] == "IRREG"]

    # Should not error; rows within threshold should have values, beyond should be NaN
    beyond = aligned[aligned["days_since_update"] > 65]
    assert beyond["value"].isna().all()


def test_missing_freq_key_falls_back_to_unknown():
    """If native_freq not in max_staleness_days, falls back to 'unknown' key."""
    df = _make_monthly_series(start="2020-01-02", periods=1)
    cal = _calendar("2020-01-01", "2020-06-01")

    with_staleness = add_staleness_indicators(df, cal)
    # 'monthly' key is absent; should use 'unknown'=30
    aligned = align_to_calendar(
        with_staleness, cal, max_staleness_days={"unknown": 30}
    )
    aligned = aligned[aligned["series_code"] == "TEST"]
    beyond = aligned[aligned["days_since_update"] > 30]
    assert beyond["value"].isna().all(), "Should fall back to 'unknown' threshold"


# ---------------------------------------------------------------------------
# Multiple series — each uses its own frequency threshold
# ---------------------------------------------------------------------------

def test_multiple_series_independent_thresholds():
    """Daily and monthly series in same call each get their own threshold."""
    # Monthly series: 1 observation (only 1 row → infer_freq returns "unknown"; use unknown=65)
    df_monthly = _make_monthly_series(start="2020-01-02", periods=1, series_code="MONTHLY")

    # Daily series: 20 closely-spaced observations so infer_freq → "daily".
    # After the 20th observation, no more updates → gap grows beyond threshold=3.
    df_daily = _make_daily_series(start="2020-01-02", periods=20, series_code="DAILY")

    df = pd.concat([df_monthly, df_daily], ignore_index=True)
    cal = _calendar("2020-01-01", "2020-05-01")

    with_staleness = add_staleness_indicators(df, cal)
    aligned = align_to_calendar(
        with_staleness, cal,
        max_staleness_days={"daily": 3, "monthly": 65, "unknown": 65}
    )

    # Monthly series (inferred as unknown; threshold=65): values within 65 days kept
    monthly_alive = aligned[
        (aligned["series_code"] == "MONTHLY") & (aligned["days_since_update"] <= 65)
    ]
    assert (monthly_alive["value"].dropna() == 1.0).all()

    # Daily series: rows past day 3 with no new update must be NaN
    daily_dead = aligned[
        (aligned["series_code"] == "DAILY") & (aligned["days_since_update"] > 3)
    ]
    assert daily_dead["value"].isna().all()
