"""Macro data validation functions."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def validate_data(
    df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
    expected_calendar: Optional[pd.DatetimeIndex] = None,
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
        "series_details": {},
    }

    # ========================================
    # 1. SCHEMA VALIDATION
    # ========================================
    logger.debug("Validating schema...")
    required_columns = ["series_code", "date", "value", "series_name", "category"]
    missing_cols = set(required_columns) - set(df.columns)

    if missing_cols:
        report["issues"].append("Missing required columns: %s" % missing_cols)
        report["status"] = "FAIL"

    # Check for staleness indicators
    has_staleness = all(
        col in df.columns for col in ["is_new_data", "days_since_update", "native_freq"]
    )
    report["statistics"]["has_staleness_indicators"] = has_staleness

    if not has_staleness:
        report["warnings"].append(
            "Staleness indicators not found (is_new_data, days_since_update, native_freq)"
        )

    # ========================================
    # 2. DATA TYPE VALIDATION
    # ========================================
    logger.debug("Validating data types...")

    # Check date is datetime
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        report["issues"].append(
            "'date' column is %s, should be datetime64[ns]" % df["date"].dtype
        )
        report["status"] = "FAIL"

    # Check value is numeric
    if "value" in df.columns and not pd.api.types.is_numeric_dtype(df["value"]):
        report["issues"].append(
            "'value' column is %s, should be numeric" % df["value"].dtype
        )
        report["status"] = "FAIL"

    # ========================================
    # 3. COMPLETENESS CHECKS
    # ========================================
    logger.debug("Checking completeness...")

    # Count nulls per column
    null_counts = df.isnull().sum()
    report["statistics"]["null_counts"] = null_counts.to_dict()

    # Critical nulls in identifier columns (series_code, date must never be null)
    id_nulls = null_counts[["series_code", "date"]].sum()
    if id_nulls > 0:
        report["issues"].append(
            "Found %d null values in identifier columns (series_code, date)" % id_nulls
        )
        report["status"] = "FAIL"

    # Value nulls: critical only when staleness tracking is absent.
    # With staleness tracking, nulls in value are intentional (max_staleness_days NaN-out).
    value_nulls = int(null_counts["value"])
    if value_nulls > 0:
        if has_staleness:
            report["warnings"].append(
                "Found %d null values in 'value' column — expected from staleness NaN-out"
                % value_nulls
            )
        else:
            report["issues"].append(
                "Found %d null values in 'value' column" % value_nulls
            )
            report["status"] = "FAIL"

    # Series count
    series_count = df["series_code"].nunique()
    report["statistics"]["series_count"] = series_count

    # Check against config if provided
    if cfg and "series" in cfg:
        expected_series = set(s["id"] for s in cfg["series"].values())
        actual_series = set(df["series_code"].unique())

        missing_series = expected_series - actual_series
        unexpected_series = actual_series - expected_series

        if missing_series:
            report["warnings"].append("Missing expected series: %s" % missing_series)

        if unexpected_series:
            report["warnings"].append("Unexpected series found: %s" % unexpected_series)

    # ========================================
    # 4. TEMPORAL CONSISTENCY
    # ========================================
    logger.debug("Validating temporal consistency...")

    # Date range
    if "date" in df.columns and len(df) > 0:
        date_min = df["date"].min()
        date_max = df["date"].max()
        report["statistics"]["date_range"] = {
            "start": date_min.strftime("%Y-%m-%d"),
            "end": date_max.strftime("%Y-%m-%d"),
            "days": (date_max - date_min).days,
        }

        # Check for future dates
        today = pd.Timestamp.today()
        future_dates = df[df["date"] > today]
        if len(future_dates) > 0:
            report["warnings"].append(
                "Found %d observations with future dates" % len(future_dates)
            )

    # Check for duplicates
    duplicates = df.duplicated(subset=["series_code", "date"]).sum()
    report["statistics"]["duplicate_date_series_pairs"] = duplicates

    if duplicates > 0:
        report["issues"].append(
            "Found %d duplicate (series_code, date) pairs" % duplicates
        )
        report["status"] = "FAIL"

    # Check sorting
    is_sorted = df[["series_code", "date"]].equals(
        df[["series_code", "date"]].sort_values(["series_code", "date"]).reset_index(drop=True)  # type: ignore[index]  # pandas boolean/multi-col indexing — mypy cannot narrow DataFrame.__getitem__
    )
    report["statistics"]["is_sorted"] = is_sorted

    if not is_sorted:
        report["warnings"].append("Data is not sorted by (series_code, date)")

    # ========================================
    # 5. VALUE QUALITY CHECKS
    # ========================================
    logger.debug("Validating value quality...")

    if "value" in df.columns:
        # Check for infinities
        inf_count = np.isinf(df["value"]).sum()
        report["statistics"]["infinite_values"] = inf_count

        if inf_count > 0:
            report["issues"].append("Found %d infinite values" % inf_count)
            report["status"] = "FAIL"

        # Basic statistics
        report["statistics"]["value_stats"] = {
            "mean": float(df["value"].mean()),
            "std": float(df["value"].std()),
            "min": float(df["value"].min()),
            "max": float(df["value"].max()),
            "median": float(df["value"].median()),
        }

    # ========================================
    # 6. SERIES-SPECIFIC VALIDATION
    # ========================================
    logger.debug("Validating individual series...")

    for series_code in df["series_code"].unique():
        series_df = df[df["series_code"] == series_code]

        series_report = {
            "observations": len(series_df),
            "date_range": {
                "start": series_df["date"].min().strftime("%Y-%m-%d"),
                "end": series_df["date"].max().strftime("%Y-%m-%d"),
            },
            "null_values": int(series_df["value"].isnull().sum()),  # type: ignore[call-overload]  # pandas Series.isnull().sum() — mypy cannot narrow int() conversion
            "value_range": {
                "min": float(series_df["value"].min()),
                "max": float(series_df["value"].max()),
            },
        }

        # Check if has staleness info
        if has_staleness and "is_new_data" in series_df.columns:
            new_data_count = series_df["is_new_data"].sum()
            series_report["new_data_points"] = int(new_data_count)
            series_report["forward_filled_points"] = len(series_df) - int(
                new_data_count
            )

            if "native_frequency" in series_df.columns:
                series_report["native_frequency"] = series_df["native_frequency"].iloc[0]  # type: ignore[index]  # pandas Series.iloc[0] — mypy cannot infer scalar return type

        # Check for constant values (zero variance)
        if series_df["value"].std() == 0:
            report["warnings"].append(
                "%s: Zero variance (constant values)" % series_code
            )

        # Check for extreme outliers (>5 std devs)
        mean = series_df["value"].mean()
        std = series_df["value"].std()
        if std > 0:
            outliers = np.abs(series_df["value"] - mean) > 5 * std
            outlier_count = outliers.sum()
            if outlier_count > 0:
                series_report["potential_outliers"] = int(outlier_count)
                report["warnings"].append(
                    "%s: %d potential outliers (>5σ)" % (series_code, outlier_count)
                )

        report["series_details"][series_code] = series_report

    # ========================================
    # 7. CALENDAR ALIGNMENT (if expected calendar provided)
    # ========================================
    if expected_calendar is not None:
        logger.debug("Validating calendar alignment...")

        # Scope calendar check to the data's actual date range.
        # trim_to_common_start legitimately removes early calendar dates, so comparing
        # against the full master calendar will always produce spurious "missing dates".
        data_start = df["date"].min()
        data_end = df["date"].max()
        scoped_calendar = expected_calendar[
            (expected_calendar >= data_start) & (expected_calendar <= data_end)
        ]
        expected_dates = set(scoped_calendar)
        actual_dates = set(df["date"].unique())

        missing_dates = expected_dates - actual_dates
        extra_dates = actual_dates - expected_dates

        if missing_dates:
            report["warnings"].append(
                "Missing %d business days within data range (%s to %s)"
                % (
                    len(missing_dates),
                    data_start.strftime("%Y-%m-%d"),
                    data_end.strftime("%Y-%m-%d"),
                )
            )
            report["statistics"]["missing_calendar_dates"] = len(missing_dates)

        if extra_dates:
            report["warnings"].append(
                "Found %d dates not in expected calendar" % len(extra_dates)
            )
            report["statistics"]["extra_calendar_dates"] = len(extra_dates)

    # ========================================
    # 8. STALENESS VALIDATION (if present)
    # ========================================
    if has_staleness:
        logger.debug("Validating staleness indicators...")

        # Check is_new_data is boolean or null
        if "is_new_data" in df.columns:
            new_data_total = df["is_new_data"].sum()
            report["statistics"]["total_new_data_points"] = int(new_data_total)
            report["statistics"]["total_forward_filled_points"] = len(df) - int(
                new_data_total
            )

        # Check days_since_update is non-negative
        if "days_since_update" in df.columns:
            negative_staleness = (df["days_since_update"] < 0).sum()
            if negative_staleness > 0:
                report["issues"].append(
                    "Found %d negative days_since_update values" % negative_staleness
                )
                report["status"] = "FAIL"

            max_staleness = df.groupby("series_code")["days_since_update"].max()
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
    Log a validation report at INFO level.

    Args:
        report: Validation report dict from validate_data()
    """
    logger.info("VALIDATION REPORT")
    logger.info("-" * 100)

    # Status
    logger.info("Status: %s", report["status"])
    logger.info("Timestamp: %s", report["timestamp"])

    # Issues
    if report["issues"]:
        logger.warning("CRITICAL ISSUES (%d):", len(report["issues"]))
        for issue in report["issues"]:
            logger.warning("  \u2022 %s", issue)

    # Warnings
    if report["warnings"]:
        logger.info("WARNINGS (%d):", len(report["warnings"]))
        for warning in report["warnings"]:
            logger.info("  - %s", warning)

    # Key Statistics
    logger.info("KEY STATISTICS:")
    stats = report["statistics"]
    logger.info("  Total rows: %s", "{:,}".format(stats.get("total_rows", 0)))
    logger.info("  Series count: %s", stats.get("series_count", "N/A"))

    if "date_range" in stats:
        dr = stats["date_range"]
        logger.info(
            "  Date range: %s to %s (%d days)", dr["start"], dr["end"], dr["days"]
        )

    if stats.get("has_staleness_indicators"):
        new_pts = stats.get("total_new_data_points", 0)
        ffill_pts = stats.get("total_forward_filled_points", 0)
        total = new_pts + ffill_pts
        pct_ffill = (ffill_pts / total * 100) if total > 0 else 0
        logger.info(
            "  Staleness: %d new points, %d forward-filled (%.1f%%)",
            new_pts,
            ffill_pts,
            pct_ffill,
        )

    # Series Summary Table
    logger.info("SERIES SUMMARY:")
    logger.info("  %-10s %8s %24s %24s", "Code", "Obs", "Date Range", "Value Range")
    logger.info("  %s %s %s %s", "-" * 10, "-" * 8, "-" * 24, "-" * 24)

    for series_code in sorted(report["series_details"].keys()):
        details = report["series_details"][series_code]
        obs = details["observations"]
        date_start = details["date_range"]["start"]
        date_end = details["date_range"]["end"]
        val_min = details["value_range"]["min"]
        val_max = details["value_range"]["max"]

        date_range_str = "%s to %s" % (date_start, date_end[-5:])
        value_range_str = "[%8.2f, %8.2f]" % (val_min, val_max)

        logger.info(
            "  %-10s %8s %24s %24s",
            series_code,
            "{:,}".format(obs),
            date_range_str,
            value_range_str,
        )

    logger.info("-" * 100)
