"""Calendar utilities for creating master business day calendars."""

import pandas as pd


def create_master_calendar(start_date: str, end_date: str | None = None) -> pd.DatetimeIndex:
    """
    Create a master calendar of business days between start and end dates.
    
    Uses US business days (excluding weekends and US federal holidays).
    
    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format). If None, uses today.
    
    Returns:
        pd.DatetimeIndex: DatetimeIndex of business days
    
    Examples:
        >>> calendar = create_master_calendar("2000-01-01", "2024-12-31")
        >>> len(calendar)
        6290
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
