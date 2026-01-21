"""Macro data cleaning utilities."""

import pandas as pd


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
    first_dates = df[df['value'].notna()].groupby('series_code')['date'].min()
    
    # Get the latest of these first dates
    latest_start_date = first_dates.max()
    
    print(f"  Trimming to common start date: {latest_start_date.date()}")
    
    # Filter to start from that date
    result: pd.DataFrame = df[df['date'] >= latest_start_date].reset_index(drop=True) # type: ignore
    
    rows_removed = len(df) - len(result)
    print(f"  Removed {int(rows_removed/df['series_code'].nunique())} dates per series")
    
    return result
