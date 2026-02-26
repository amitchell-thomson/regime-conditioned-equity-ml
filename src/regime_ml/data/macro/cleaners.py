"""Macro data cleaning utilities."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def roll_weekend_releases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Move weekend dates to next business day.
    Represents when you could actually trade on the information.

    Args:
        df: DataFrame with 'date' column

    Returns:
        pd.DataFrame: The dataframe with weekend dates rolled to next business day.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    from pandas.tseries.offsets import BDay

    df["date"] = df["date"].apply(
        lambda x: x + BDay(0) if x.weekday() < 5 else x + BDay(1)
    )

    # BDay(0) returns the date itself if business day
    # BDay(1) rolls forward to next business day if weekend

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw macro data: types, weekends, duplicates.
    NO Forward filling or calendar alignment.

    Args:
        df: Raw macro dataframe

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """

    # 1. Type Conversions
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 2. Weekend Roll
    df = roll_weekend_releases(df)

    # 3. Duplicate Removal
    df = df.sort_values(by=["series_code", "date"])
    df = df.drop_duplicates(subset=["series_code", "date"], keep="last")

    # 4. Sort
    df = df.sort_values(by=["series_code", "date"]).reset_index(drop=True)

    return df


def trim_to_own_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim each series to its own first valid observation, allowing ragged starts.

    Rows where 'value' is NaN before a series' first real observation are dropped
    per-series. Different series may therefore start on different dates in the
    returned DataFrame (ragged). This is the correct behaviour when mixing FRED
    and ALFRED sources: FRED market-data series start in 2000 while ALFRED
    economic series start when ALFRED began archiving them (~2009-2011).

    Args:
        df: DataFrame with 'series_code', 'date', and 'value' columns.

    Returns:
        pd.DataFrame: Long-format DataFrame with per-series leading NaN rows removed.
    """
    first_valid = df[df["value"].notna()].groupby("series_code")["date"].min()
    df_with_start = df.join(first_valid.rename("_series_start"), on="series_code")
    result = df_with_start[df_with_start["date"] >= df_with_start["_series_start"]]
    result = result.drop(columns="_series_start").reset_index(drop=True)
    n_removed = len(df) - len(result)
    logger.info(
        "trim_to_own_start: removed %d pre-start rows; series start dates:",
        n_removed,
    )
    for code, start in first_valid.sort_values().items():
        logger.info("  %-10s  %s", code, start.date())
    return result


def trim_to_common_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim all series to start at the latest start date across all series.
    Ensures all series have aligned start date.

    Args:
        df: DataFrame with 'date', 'series_code', and 'value' columns

    Returns:
        pd.DataFrame: Trimmed dataframe
    """
    # Find first valid date for each series
    first_dates = df[df["value"].notna()].groupby("series_code")["date"].min()

    # Get the latest of these first dates
    latest_start_date = first_dates.max()

    logger.info("Trimming to common start date: %s", latest_start_date.date())

    # Filter to start from that date
    result: pd.DataFrame = df[df["date"] >= latest_start_date].reset_index(drop=True)  # type: ignore[index]  # pandas boolean indexing — mypy cannot narrow DataFrame.__getitem__ from Series[bool]

    rows_removed = len(df) - len(result)
    logger.info(
        "Removed %d dates per series", int(rows_removed / df["series_code"].nunique())
    )

    return result
