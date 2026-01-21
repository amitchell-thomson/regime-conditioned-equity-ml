"""Macro data alignment utilities for calendar alignment and staleness tracking."""

import pandas as pd
from tqdm import tqdm


def add_staleness_indicators(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Add metadata about the data freshness before forward-filling.
    This preserves the information about when the data was actually released.
    
    Args:
        df: Cleaned macro dataframe
        calendar: Master calendar (not used directly, kept for API consistency)
    
    Returns:
        pd.DataFrame: Dataframe with staleness indicators added
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

    # 3. Add observation sequence number (at a native frequency)
    df["obs_number"] = df.groupby("series_code").cumcount() + 1

    return df


def align_to_calendar(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align dataframe to master calendar by forward-filling missing dates.
    Staleness indicators are preserved.
    
    Args:
        df: Macro dataframe with staleness indicators
        calendar: Master business day calendar
    
    Returns:
        pd.DataFrame: Calendar-aligned dataframe with forward-filled values
    """
    aligned_series = []
    
    for series_code in tqdm(df['series_code'].unique(), desc="Aligning series"):
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
