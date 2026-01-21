"""Helper functions to clean and preprocess data."""

import pandas as pd

def roll_weekend_releases(df: pd.DataFrame):
    """
    Move weekend dates to next business day.
    Represents when you could actually trade on the information.

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
    Clean raw data: types, weekends, duplicates.
    NO Forward filling or calendar alignment

    Returns
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