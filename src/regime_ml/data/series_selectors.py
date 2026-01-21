"""Data selectors for the regime-conditioned equity ML project."""

from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd

from regime_ml.utils import load_configs


def select_data(
    raw_data: pd.DataFrame,
    cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Select specific series from raw data based on configuration.
    
    This function filters raw data to only include the series specified
    in the configuration file. It uses the series IDs from the config to match
    against the 'series_code' column in the raw data.
    
    Args:
        raw_data: Raw dataframe containing all data.
                 Expected columns: series_code, date, value, series_name, category
        cfg: Configuration dictionary containing:
            - 'series': Dict of series definitions, each with an 'id' field
            - 'raw_path': Path where selected data should be saved (optional)
    
    Returns:
        pd.DataFrame: Filtered dataframe containing only the specified series,
                     preserving the original column structure and ordering.
    
    Raises:
        KeyError: If required configuration keys are missing.
        ValueError: If no series are found in the configuration.
        
    Note:
        The function extracts unique series IDs from the configuration to avoid
        duplicates, then filters the raw data to include only those series.
    """
    # Extract unique series IDs from configuration
    # Using a set comprehension ensures we only query each series once
    tickers = list({value["id"] for key, value in cfg["series"].items()})
    
    if not tickers:
        raise ValueError("No series found in configuration")
    
    # Filter raw data to only include configured series
    # This is more efficient than concatenating multiple filtered dataframes
    df_selected: pd.DataFrame = raw_data[raw_data["series_code"].isin(tickers)].copy()  # type: ignore
    
    # Sort by series code and date for consistency
    if "date" in df_selected.columns:
        df_selected = df_selected.sort_values(by=["series_code", "date"]).reset_index(drop=True)  # type: ignore
    
    return df_selected
