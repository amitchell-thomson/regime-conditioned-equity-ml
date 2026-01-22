"""Macro data processing module."""

from .loaders import load_raw_data
from .cleaners import clean_data, roll_weekend_releases, trim_to_common_start
from .alignment import add_staleness_indicators, align_to_calendar
from .selector import select_data
from .validators import validate_data, print_validation_report
from .pipeline import run_macro_data_pipeline

__all__ = [
    # Loaders
    'load_raw_data',
    
    # Cleaners
    'clean_data',
    'roll_weekend_releases',
    'trim_to_common_start',
    
    # Alignment
    'add_staleness_indicators',
    'align_to_calendar',
    
    # Selectors
    'select_data',
    
    # Validators
    'validate_data',
    'print_validation_report',
    
    # Pipeline
    'run_macro_data_pipeline',
]
