"""Data processing module for regime-conditioned equity ML project.

This module is organized into:
- common: Shared utilities for all data types
- macro: Macro time series data processing
- equity: Equity price data processing (future)
"""

# Common utilities
from .common import (
    get_project_root,
    load_dataframe,
    create_master_calendar,
)

# Macro data utilities
from .macro import (
    load_raw_data,
    select_data,
    clean_data,
    roll_weekend_releases,
    add_staleness_indicators,
    align_to_calendar,
    trim_to_common_start,
    validate_data,
    print_validation_report,
    run_macro_data_pipeline,
)

# Equity utilities (to be added later)
# from .equity import ...

__all__ = [
    # Common
    'get_project_root',
    'load_dataframe',
    'create_master_calendar',
    
    # Macro
    'load_raw_data',
    'select_data',
    'clean_data',
    'roll_weekend_releases',
    'add_staleness_indicators',
    'align_to_calendar',
    'trim_to_common_start',
    'validate_data',
    'print_validation_report',
    'run_macro_data_pipeline',
]
