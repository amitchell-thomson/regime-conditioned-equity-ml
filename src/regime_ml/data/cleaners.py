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

def create_master_calendar(start_date: str, end_date: str | None = None) -> pd.DatetimeIndex:
    """
    Create a master calendar of business days between start and end dates.
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    if end_date is None:
        end_date = str(pd.Timestamp.today())

    master_calendar = pd.date_range(
        start=start_date,
        end=end_date,
        freq=us_bd
    )
    return master_calendar

def add_staleness_indicators(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Add metadata about the data freshness before forward-filling.
    This preserves the information about when the data was actually released
    """
    df = df.copy()
    df = df.sort_values(by=["series_code", "date"])

    # 1. Mark when the value actually changed (new datapoint)
    df["is_new_data"] = (
        df.groupby("series_code")["value"]
        .transform(lambda x: x != x.shift(1))
        .fillna(True)
    )

    # 2. Detect native frequency (daily, weekly, monthly, etc.)
    def infer_freq(dates):
        if len(dates) < 2:
            return 'unknown'
        median_gap = dates.diff().median().days
        if median_gap <= 1:
            return 'daily'
        elif median_gap <= 7:
            return 'weekly'
        elif median_gap <= 31:
            return 'monthly'
        else:
            return 'irregular'
    
    df["native_freq"] = df.groupby("series_code")["date"].transform(lambda x: infer_freq(x))

    # 3. Add observation sequency number (at a native frequency)
    df["obs_number"] =df.groupby("series_code").cumcount() + 1

    return df

def align_to_calendar(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align dataframe to master calendar by forward-filling missing dates.
    Staleness indicators are preserved.
    """
    aligned_series = []
    
    for series_code in df['series_code'].unique():
        series_df = df[df['series_code'] == series_code].set_index('date')
        
        # Reindex to master calendar
        aligned = series_df.reindex(calendar)
        
        # Forward-fill values AND metadata
        aligned['value'] = aligned['value'].ffill()
        aligned['series_code'] = series_code
        aligned['series_name'] = aligned['series_name'].ffill()
        aligned['category'] = aligned['category'].ffill()
        aligned['native_freq'] = aligned['native_freq'].ffill()
        aligned['obs_number'] = aligned['obs_number'].ffill()
        
        # is_new_data stays as-is (True where data updated, NaN elsewhere)
        # This is the key: you can see which days had real updates!
        
        # Add days since last update
        last_update_date = aligned[aligned['is_new_data'] == True].index
        aligned['days_since_update'] = 0
        for date in aligned.index:
            days_since = (date - last_update_date[last_update_date <= date].max()).days # type: ignore
            aligned.loc[date, 'days_since_update'] = days_since
        
        aligned_series.append(aligned)
    
    df_final = pd.concat(aligned_series)
    return df_final.reset_index().rename(columns={'index': 'date'})