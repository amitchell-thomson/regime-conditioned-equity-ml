"""Macro data alignment utilities for calendar alignment and staleness tracking."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_staleness_indicators(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Add metadata about the data freshness before forward-filling.
    This preserves the information about when the data was actually released.

    Every row in the pre-alignment DataFrame represents a real publication event,
    so is_new_data is unconditionally True here. After align_to_calendar() reindexes
    to the master calendar, forward-filled dates receive is_new_data=NaN (from
    reindex), correctly distinguishing them from real observations.

    This replaces the previous value-equality check, which incorrectly flagged
    is_new_data=False when a series published the same value two periods in a row,
    and incorrectly flagged is_new_data=True for FRED revisions.

    Args:
        df: Cleaned macro dataframe (pre-alignment, one row per real observation)
        calendar: Master calendar (not used directly, kept for API consistency)

    Returns:
        pd.DataFrame: Dataframe with staleness indicators added
    """
    df = df.copy()
    df = df.sort_values(by=["series_code", "date"])

    # Every pre-alignment row is a real publication event — is_new_data is unconditionally True.
    # align_to_calendar() sets is_new_data=NaN for forward-filled calendar dates via reindex.
    df["is_new_data"] = True

    # Detect native frequency (daily, weekly, monthly, irregular)
    def infer_freq(dates: pd.Series) -> str:
        if len(dates) < 2:
            return "unknown"
        median_gap = dates.diff().median().days
        if median_gap <= 1:
            return "daily"
        elif median_gap <= 7:
            return "weekly"
        elif median_gap <= 31:
            return "monthly"
        return "irregular"

    df["native_freq"] = df.groupby("series_code")["date"].transform(lambda x: infer_freq(x))

    # Add observation sequence number (at native frequency)
    df["obs_number"] = df.groupby("series_code").cumcount() + 1

    return df


def align_to_calendar(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    max_staleness_days: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Align dataframe to master calendar by forward-filling missing dates.
    Staleness indicators are preserved.

    Args:
        df: Macro dataframe with staleness indicators
        calendar: Master business day calendar
        max_staleness_days: Optional dict mapping native_freq → max days before NaN-out.
                            Example: {"daily": 5, "weekly": 14, "monthly": 65}.
                            If None, forward-fill runs with no limit (original behavior).

    Returns:
        pd.DataFrame: Calendar-aligned dataframe with forward-filled values
    """
    aligned_series = []

    all_series = df["series_code"].unique()
    logger.info("Aligning %d series to calendar...", len(all_series))
    for series_code in all_series:
        series_df = df[df["series_code"] == series_code].set_index("date")

        # Reindex to master calendar
        aligned = series_df.reindex(calendar)

        # Forward-fill values AND metadata
        aligned["value"] = aligned["value"].ffill()
        aligned["series_code"] = series_code
        aligned["series_name"] = aligned["series_name"].ffill()
        aligned["category"] = aligned["category"].ffill()
        aligned["native_freq"] = aligned["native_freq"].ffill()
        aligned["obs_number"] = aligned["obs_number"].ffill()

        # is_new_data stays as-is (True where data updated, NaN elsewhere)
        # This is the key: you can see which days had real updates!

        # Add days since last update (vectorized: O(T) instead of O(T²))
        # Each run of forward-filled rows forms a group; cumcount gives elapsed days.
        is_new = aligned["is_new_data"].notna()
        group_id = is_new.cumsum()  # increments at every real observation
        aligned["days_since_update"] = aligned.groupby(group_id).cumcount()

        # Apply max staleness limit: NaN out values that are too stale
        if max_staleness_days is not None:
            native_freq_notnull = aligned["native_freq"].dropna()
            freq = native_freq_notnull.iloc[0] if not native_freq_notnull.empty else "unknown"
            threshold = max_staleness_days.get(freq, max_staleness_days.get("unknown", 65))
            stale_mask = aligned["days_since_update"] > threshold
            if stale_mask.any():
                aligned.loc[stale_mask, "value"] = np.nan
                logger.debug(
                    "Series %s: NaN-out %d excessively stale rows (freq=%s, threshold=%d days)",
                    series_code,
                    int(stale_mask.sum()),
                    freq,
                    threshold,
                )

        aligned_series.append(aligned)

    df_final = pd.concat(aligned_series)
    return df_final.reset_index().rename(columns={"index": "date"})


def build_realtime_series(
    alfred_df: pd.DataFrame,
    series_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reconstruct a point-in-time (real-time) time series from ALFRED vintage data.

    For each publication date (realtime_start), determines the value that was
    actually available: among all observations published on or before that date,
    takes the most recent observation's value.

    This eliminates lookahead bias from publication lags: the reconstructed series
    reflects only what was known at each realtime_start, not final revised values.

    ALFRED parquet columns expected:
        series_code, date (obs period), realtime_start (pub date),
        realtime_end, value, series_name, category

    Args:
        alfred_df: ALFRED data with the columns listed above.
        series_codes: Optional list of series to process. If None, all are processed.

    Returns:
        pd.DataFrame: Point-in-time series with same columns as FRED output:
            series_code, date, value, series_name, category
            where date = realtime_start (the causal date when value became available).

    Notes:
        - Output 'date' column contains realtime_start values, not obs_date values.
          CFNAI for January 2020 will appear on ~2020-02-28 (its publication date).
        - This allows drop-in replacement into the existing pipeline unchanged.
        - All output rows have is_new_data=True after add_staleness_indicators()
          (consistent with the 2.1 fix: every pre-alignment row is a real event).
    """
    alfred_df = alfred_df.copy()
    alfred_df["date"] = pd.to_datetime(alfred_df["date"])
    alfred_df["realtime_start"] = pd.to_datetime(alfred_df["realtime_start"])

    if series_codes is not None:
        alfred_df = alfred_df[alfred_df["series_code"].isin(series_codes)]

    if alfred_df.empty:
        raise ValueError("build_realtime_series: no data after filtering to series_codes=%s" % series_codes)

    results = []
    for code, group in alfred_df.groupby("series_code"):
        group = group.sort_values(["realtime_start", "date"])
        rows = []

        for pub_date, _ in group.groupby("realtime_start"):
            # All vintages available as of this publication date
            available = group[group["realtime_start"] <= pub_date]
            if available.empty:
                continue
            # Use the most recent observation period (max obs_date) available at pub_date.
            # Among multiple vintages of that obs_date, take the latest revision (max realtime_start).
            best_obs_date = available["date"].max()
            candidates = available[available["date"] == best_obs_date]
            best_row = candidates.loc[candidates["realtime_start"].idxmax()]
            rows.append(
                {
                    "series_code": code,
                    "date": pub_date,  # causal date = publication date
                    "value": best_row["value"],
                    "series_name": best_row["series_name"],
                    "category": best_row["category"],
                }
            )

        logger.debug(
            "build_realtime_series: %s -> %d point-in-time observations",
            code,
            len(rows),
        )
        if rows:
            results.append(pd.DataFrame(rows))

    if not results:
        raise ValueError("build_realtime_series produced no output. " "Check series_codes and input data.")

    output = pd.concat(results, ignore_index=True)
    logger.info(
        "build_realtime_series: reconstructed %d point-in-time rows for %d series",
        len(output),
        output["series_code"].nunique(),
    )
    return output
