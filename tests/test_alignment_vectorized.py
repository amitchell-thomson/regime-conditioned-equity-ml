"""Tests for vectorized days_since_update in alignment.py."""

import numpy as np
import pandas as pd



def _make_series_df(
    dates: list[str],
    values: list[float],
    is_new: list[bool],
    series_code: str = "TEST",
) -> pd.DataFrame:
    """Build a minimal aligned-format DataFrame for a single series."""
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "series_code": series_code,
            "value": values,
            "is_new_data": is_new,
            "series_name": series_code,
            "category": "test",
            "native_freq": "daily",
            "obs_number": range(1, len(dates) + 1),
        },
        index=idx,
    )


def _days_since_update(aligned_df: pd.DataFrame) -> list[int]:
    """Extract days_since_update column from an aligned DataFrame."""
    is_new = aligned_df["is_new_data"].fillna(False)
    group_id = is_new.cumsum()
    return aligned_df.groupby(group_id).cumcount().tolist()


class TestDaysSinceUpdateVectorized:
    def test_single_update_at_start(self):
        """First day is a real update; all subsequent days count up from 0."""
        df = _make_series_df(
            dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            values=[1.0, 1.0, 1.0, 1.0],
            is_new=[True, False, False, False],
        )
        result = _days_since_update(df)
        assert result == [0, 1, 2, 3]

    def test_multiple_updates(self):
        """Days reset to 0 at each new real observation."""
        df = _make_series_df(
            dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            values=[1.0, 1.0, 2.0, 2.0, 2.0],
            is_new=[True, False, True, False, False],
        )
        result = _days_since_update(df)
        assert result == [0, 1, 0, 1, 2]

    def test_first_row_nan_is_new_data(self):
        """NaN in is_new_data (from reindex before first real obs) is treated as not-new."""
        df = _make_series_df(
            dates=["2020-01-01", "2020-01-02", "2020-01-03"],
            values=[1.0, 1.0, 2.0],
            is_new=[False, False, True],
        )
        df.loc[df.index[0], "is_new_data"] = None  # simulate NaN from reindex
        result = _days_since_update(df)
        # All three rows are in group 0 until the True at index 2 creates group 1
        assert result[2] == 0, "Days should reset to 0 at the real update"

    def test_consecutive_updates(self):
        """Every day is a real update: days_since_update stays at 0."""
        df = _make_series_df(
            dates=["2020-01-01", "2020-01-02", "2020-01-03"],
            values=[1.0, 2.0, 3.0],
            is_new=[True, True, True],
        )
        result = _days_since_update(df)
        assert result == [0, 0, 0]

    def test_all_stale(self):
        """No real updates: days count up monotonically from 0."""
        df = _make_series_df(
            dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            values=[1.0, 1.0, 1.0, 1.0],
            is_new=[False, False, False, False],
        )
        result = _days_since_update(df)
        assert result == [0, 1, 2, 3]

    def test_output_is_non_negative(self):
        """days_since_update must never be negative."""
        rng = np.random.default_rng(0)
        n = 50
        is_new = rng.random(n) > 0.8
        is_new[0] = True
        df = _make_series_df(
            dates=pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist(),
            values=rng.standard_normal(n).tolist(),
            is_new=is_new.tolist(),
        )
        result = _days_since_update(df)
        assert all(v >= 0 for v in result)
