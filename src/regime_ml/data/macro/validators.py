"""Macro data validation functions."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def validate_data(
    df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
    expected_calendar: Optional[pd.DatetimeIndex] = None
) -> Dict[str, Any]:
    """
    Comprehensive validation of processed macro dataframe.
    
    Validates:
    - Schema and data types
    - Completeness (missing values, coverage)
    - Temporal consistency (ordering, duplicates, gaps)
    - Value quality (ranges, outliers, infinities)
    - Staleness indicators (if present)
    - Alignment with expected series (if config provided)
    
    Args:
        df: Processed dataframe to validate
        cfg: Optional configuration dict with expected series
        expected_calendar: Optional expected date range (master calendar)
    
    Returns:
        Dict containing validation report with issues, warnings, and statistics
    """
    report = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PASS",  # Will change to FAIL if critical issues found
        "issues": [],
        "warnings": [],
        "statistics": {},
        "series_details": {}
    }
    
    # ========================================
    # 1. SCHEMA VALIDATION
    # ========================================
    print("  - Validating schema...")
    required_columns = ['series_code', 'date', 'value', 'series_name', 'category']
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        report["issues"].append(f"Missing required columns: {missing_cols}")
        report["status"] = "FAIL"
    
    # Check for staleness indicators
    has_staleness = all(col in df.columns for col in ['is_new_data', 'days_since_update', 'native_freq'])
    report["statistics"]["has_staleness_indicators"] = has_staleness
    
    if not has_staleness:
        report["warnings"].append("Staleness indicators not found (is_new_data, days_since_update, native_freq)")
    
    # ========================================
    # 2. DATA TYPE VALIDATION
    # ========================================
    print("  - Validating data types...")
    
    # Check date is datetime
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        report["issues"].append(f"'date' column is {df['date'].dtype}, should be datetime64[ns]")
        report["status"] = "FAIL"
    
    # Check value is numeric
    if 'value' in df.columns and not pd.api.types.is_numeric_dtype(df['value']):
        report["issues"].append(f"'value' column is {df['value'].dtype}, should be numeric")
        report["status"] = "FAIL"
    
    # ========================================
    # 3. COMPLETENESS CHECKS
    # ========================================
    print("  - Checking completeness...")
    
    # Count nulls per column
    null_counts = df.isnull().sum()
    report["statistics"]["null_counts"] = null_counts.to_dict()
    
    # Critical nulls
    critical_nulls = null_counts[['series_code', 'date', 'value']].sum()
    if critical_nulls > 0:
        report["issues"].append(f"Found {critical_nulls} null values in critical columns (series_code, date, value)")
        report["status"] = "FAIL"
    
    # Series count
    series_count = df['series_code'].nunique()
    report["statistics"]["series_count"] = series_count
    
    # Check against config if provided
    if cfg and 'series' in cfg:
        expected_series = set(s['id'] for s in cfg['series'].values())
        actual_series = set(df['series_code'].unique())
        
        missing_series = expected_series - actual_series
        unexpected_series = actual_series - expected_series
        
        if missing_series:
            report["warnings"].append(f"Missing expected series: {missing_series}")
        
        if unexpected_series:
            report["warnings"].append(f"Unexpected series found: {unexpected_series}")
    
    # ========================================
    # 4. TEMPORAL CONSISTENCY
    # ========================================
    print("  - Validating temporal consistency...")
    
    # Date range
    if 'date' in df.columns and len(df) > 0:
        date_min = df['date'].min()
        date_max = df['date'].max()
        report["statistics"]["date_range"] = {
            "start": date_min.strftime("%Y-%m-%d"),
            "end": date_max.strftime("%Y-%m-%d"),
            "days": (date_max - date_min).days
        }
        
        # Check for future dates
        today = pd.Timestamp.today()
        future_dates = df[df['date'] > today]
        if len(future_dates) > 0:
            report["warnings"].append(f"Found {len(future_dates)} observations with future dates")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['series_code', 'date']).sum()
    report["statistics"]["duplicate_date_series_pairs"] = duplicates
    
    if duplicates > 0:
        report["issues"].append(f"Found {duplicates} duplicate (series_code, date) pairs")
        report["status"] = "FAIL"
    
    # Check sorting
    is_sorted = df[['series_code', 'date']].equals(
        df[['series_code', 'date']].sort_values(['series_code', 'date']).reset_index(drop=True) # type: ignore
    )
    report["statistics"]["is_sorted"] = is_sorted
    
    if not is_sorted:
        report["warnings"].append("Data is not sorted by (series_code, date)")
    
    # ========================================
    # 5. VALUE QUALITY CHECKS
    # ========================================
    print("  - Validating value quality...")
    
    if 'value' in df.columns:
        # Check for infinities
        inf_count = np.isinf(df['value']).sum()
        report["statistics"]["infinite_values"] = inf_count
        
        if inf_count > 0:
            report["issues"].append(f"Found {inf_count} infinite values")
            report["status"] = "FAIL"
        
        # Basic statistics
        report["statistics"]["value_stats"] = {
            "mean": float(df['value'].mean()),
            "std": float(df['value'].std()),
            "min": float(df['value'].min()),
            "max": float(df['value'].max()),
            "median": float(df['value'].median())
        }
    
    # ========================================
    # 6. SERIES-SPECIFIC VALIDATION
    # ========================================
    print("  - Validating individual series...")
    
    for series_code in df['series_code'].unique():
        series_df = df[df['series_code'] == series_code]
        
        series_report = {
            "observations": len(series_df),
            "date_range": {
                "start": series_df['date'].min().strftime("%Y-%m-%d"),
                "end": series_df['date'].max().strftime("%Y-%m-%d")
            },
            "null_values": int(series_df['value'].isnull().sum()), # type: ignore
            "value_range": {
                "min": float(series_df['value'].min()),
                "max": float(series_df['value'].max())
            }
        }
        
        # Check if has staleness info
        if has_staleness and 'is_new_data' in series_df.columns:
            new_data_count = series_df['is_new_data'].sum()
            series_report["new_data_points"] = int(new_data_count)
            series_report["forward_filled_points"] = len(series_df) - int(new_data_count)
            
            if 'native_frequency' in series_df.columns:
                series_report["native_frequency"] = series_df['native_frequency'].iloc[0] # type: ignore
        
        # Check for constant values (zero variance)
        if series_df['value'].std() == 0:
            report["warnings"].append(f"{series_code}: Zero variance (constant values)")
        
        # Check for extreme outliers (>5 std devs)
        mean = series_df['value'].mean()
        std = series_df['value'].std()
        if std > 0:
            outliers = np.abs(series_df['value'] - mean) > 5 * std
            outlier_count = outliers.sum()
            if outlier_count > 0:
                series_report["potential_outliers"] = int(outlier_count)
                report["warnings"].append(f"{series_code}: {outlier_count} potential outliers (>5σ)")
        
        report["series_details"][series_code] = series_report
    
    # ========================================
    # 7. CALENDAR ALIGNMENT (if expected calendar provided)
    # ========================================
    if expected_calendar is not None:
        print("  - Validating calendar alignment...")
        
        expected_dates = set(expected_calendar)
        actual_dates = set(df['date'].unique())
        
        missing_dates = expected_dates - actual_dates
        extra_dates = actual_dates - expected_dates
        
        if missing_dates:
            report["warnings"].append(f"Missing {len(missing_dates)} expected dates from calendar")
            report["statistics"]["missing_calendar_dates"] = len(missing_dates)
        
        if extra_dates:
            report["warnings"].append(f"Found {len(extra_dates)} dates not in expected calendar")
            report["statistics"]["extra_calendar_dates"] = len(extra_dates)
    
    # ========================================
    # 8. STALENESS VALIDATION (if present)
    # ========================================
    if has_staleness:
        print("  - Validating staleness indicators...")
        
        # Check is_new_data is boolean or null
        if 'is_new_data' in df.columns:
            new_data_total = df['is_new_data'].sum()
            report["statistics"]["total_new_data_points"] = int(new_data_total)
            report["statistics"]["total_forward_filled_points"] = len(df) - int(new_data_total)
        
        # Check days_since_update is non-negative
        if 'days_since_update' in df.columns:
            negative_staleness = (df['days_since_update'] < 0).sum()
            if negative_staleness > 0:
                report["issues"].append(f"Found {negative_staleness} negative days_since_update values")
                report["status"] = "FAIL"
            
            max_staleness = df.groupby('series_code')['days_since_update'].max()
            report["statistics"]["max_staleness_by_series"] = max_staleness.to_dict()
    
    # ========================================
    # 9. FINAL SUMMARY
    # ========================================
    report["statistics"]["total_rows"] = len(df)
    report["statistics"]["total_issues"] = len(report["issues"])
    report["statistics"]["total_warnings"] = len(report["warnings"])
    
    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    """
    Pretty-print a validation report.
    
    Args:
        report: Validation report dict from validate_data()
    """
    print("VALIDATION REPORT")
    print("-" * 100)
    
    # Status
    print(f"Status: {report['status']}")
    print(f"Timestamp: {report['timestamp']}")
    
    # Issues
    if report["issues"]:
        print(f"\nCRITICAL ISSUES ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"  • {issue}")
    
    # Warnings
    if report["warnings"]:
        print(f"\nWARNINGS ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    
    # Key Statistics
    print(f"\nKEY STATISTICS:")
    stats = report["statistics"]
    print(f"  Total rows: {stats.get('total_rows', 'N/A'):,}")
    print(f"  Series count: {stats.get('series_count', 'N/A')}")
    
    if 'date_range' in stats:
        dr = stats['date_range']
        print(f"  Date range: {dr['start']} to {dr['end']} ({dr['days']} days)")
    
    if stats.get('has_staleness_indicators'):
        new_pts = stats.get('total_new_data_points', 0)
        ffill_pts = stats.get('total_forward_filled_points', 0)
        total = new_pts + ffill_pts
        pct_ffill = (ffill_pts / total * 100) if total > 0 else 0
        print(f"  Staleness: {new_pts:,} new points, {ffill_pts:,} forward-filled ({pct_ffill:.1f}%)")
    
    # Series Summary Table
    print(f"\nSERIES SUMMARY:")
    print(f"  {'Code':<10} {'Obs':>8} {'Date Range':>24} {'Value Range':>24}")
    print(f"  {'-'*10} {'-'*8} {'-'*24} {'-'*24}")
    
    for series_code in sorted(report["series_details"].keys()):
        details = report["series_details"][series_code]
        obs = details['observations']
        date_start = details['date_range']['start']
        date_end = details['date_range']['end']
        val_min = details['value_range']['min']
        val_max = details['value_range']['max']
        
        date_range_str = f"{date_start} to {date_end[-5:]}"  # Just show end date without year
        value_range_str = f"[{val_min:>8.2f}, {val_max:>8.2f}]"
        
        print(f"  {series_code:<10} {obs:>8,} {date_range_str:>24} {value_range_str:>24}")
    
    print(f"\n{'-' * 100}")
