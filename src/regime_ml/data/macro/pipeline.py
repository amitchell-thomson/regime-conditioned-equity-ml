"""Macro data pipeline for complete data processing workflow."""

import logging
import time
from pathlib import Path
import pandas as pd

from regime_ml.utils import load_configs
from regime_ml.data.common import create_master_calendar
from regime_ml.data.macro import (
    load_raw_data,
    load_alfred_data,
    build_realtime_series,
    select_data,
    clean_data,
    add_staleness_indicators,
    align_to_calendar,
    trim_to_own_start,
    validate_data,
    print_validation_report,
)

logger = logging.getLogger(__name__)


def _log_stage_header(stage_num: int, total_stages: int, title: str) -> None:
    """Log a consistent stage header."""
    logger.info("[%d/%d] %s", stage_num, total_stages, title)
    logger.info("-" * 100)


def _log_summary(label: str, df: pd.DataFrame, timing: float | None = None) -> None:
    """Log concise summary statistics."""
    rows = len(df)
    series = df["series_code"].nunique()
    date_range = "%s to %s" % (
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )

    if timing is not None:
        logger.info(
            "  - %s: %d rows, %d series, %s (%.2fs)",
            label,
            rows,
            series,
            date_range,
            timing,
        )
    else:
        logger.info("  - %s: %d rows, %d series, %s", label, rows, series, date_range)


def run_macro_data_pipeline() -> pd.DataFrame:
    """
    Run the complete macro data pipeline from raw to processed data.

    Pipeline Stages:
        1. Load: Load raw data from source
        2. Select & Clean: Filter to configured series, clean and deduplicate
        3. Align: Calendar alignment with staleness tracking
        4. Validate: Comprehensive quality checks
        5. Save: Export processed data

    Returns:
        pd.DataFrame: Processed and validated dataframe
    """
    pipeline_start = time.time()

    # Header
    logger.info("=" * 100)
    logger.info("MACRO DATA PIPELINE")
    logger.info("=" * 100)

    # Load configuration
    cfg = load_configs()
    macro_cfg = cfg["macro_data"]["regime_universe"]

    # =================================================================
    # STAGE 1: LOAD
    # Hybrid FRED/ALFRED loader. Each series specifies use_alfred: true/false
    # in regime_universe.yaml. The global alfred.use_alfred is the fallback
    # for any series that does not specify its own override.
    #
    # Rationale for per-series routing:
    #   - ALFRED is essential for economically revised series (CFNAI, INDPRO,
    #     PCEPILFE, NFCI, ICSA) where initial estimates differ materially from
    #     final values. ALFRED reconstructs what was known on each publication date.
    #   - Market-quoted daily series (VIXCLS, DGS2, DGS10, T10Y3M) are never
    #     revised by FRED. ALFRED adds no bias protection and its archives for
    #     these series are incomplete (T10Y3M only from 2014-01-27), which would
    #     otherwise bind the entire dataset start date.
    # =================================================================
    _log_stage_header(1, 5, "Load")
    stage_start = time.time()

    alfred_cfg = macro_cfg.get("alfred", {})
    global_use_alfred = alfred_cfg.get("use_alfred", False)

    # Partition series by their per-series use_alfred override (global is fallback)
    fred_codes: list[str] = []
    alfred_codes: list[str] = []
    for series_cfg in macro_cfg["series"].values():
        if series_cfg.get("use_alfred", global_use_alfred):
            alfred_codes.append(series_cfg["id"])
        else:
            fred_codes.append(series_cfg["id"])

    logger.info("  ALFRED (point-in-time, %d series): %s", len(alfred_codes), alfred_codes)
    logger.info("  FRED   (final revised,  %d series): %s", len(fred_codes), fred_codes)

    parts: list[pd.DataFrame] = []

    if alfred_codes:
        raw_alfred = load_alfred_data(macro_cfg["data_path"])
        pit = build_realtime_series(raw_alfred, series_codes=alfred_codes)
        parts.append(pit)

    if fred_codes:
        raw_fred = load_raw_data(macro_cfg["data_path"], macro_cfg["raw_path"])
        parts.append(raw_fred[raw_fred["series_code"].isin(fred_codes)])

    df_raw = pd.concat(parts, ignore_index=True)
    # Normalise date column: FRED parquets store datetime.date, ALFRED stores Timestamps
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    # Per-series coverage log — shows effective point-in-time history per series
    alfred_set = set(alfred_codes)
    logger.info("  %-10s  %-6s  %s", "Series", "Source", "Date range")
    logger.info("  %s  %s  %s", "-" * 10, "-" * 6, "-" * 32)
    for code, grp in df_raw.groupby("series_code"):
        source = "ALFRED" if code in alfred_set else "FRED"
        logger.info(
            "  %-10s  %-6s  %s -> %s",
            code,
            source,
            str(pd.Timestamp(grp["date"].min()))[:10],
            str(pd.Timestamp(grp["date"].max()))[:10],
        )

    _log_summary("Loaded", df_raw, time.time() - stage_start)

    # =================================================================
    # STAGE 2: SELECT & CLEAN
    # =================================================================
    _log_stage_header(2, 5, "Select & Clean")
    stage_start = time.time()

    df_selected = select_data(df_raw, macro_cfg)
    logger.info("  - Selected %d series from config", df_selected["series_code"].nunique())

    df_clean = clean_data(df_selected)
    logger.info("  - Cleaned data (types, weekends, deduplication)")

    _log_summary("Processed", df_clean, time.time() - stage_start)

    # =================================================================
    # STAGE 3: ALIGN
    # =================================================================
    _log_stage_header(3, 5, "Align to Calendar")
    stage_start = time.time()

    # Create master business day calendar
    master_calendar = create_master_calendar(macro_cfg["lookback"]["start"], macro_cfg["lookback"]["end"])
    logger.info("  - Created calendar: %d business days", len(master_calendar))

    # Add staleness tracking
    df_stale = add_staleness_indicators(df_clean, master_calendar)
    logger.info("  - Added staleness indicators")

    # Align to calendar with forward-fill, applying max staleness limit if configured
    max_staleness_cfg = macro_cfg.get("max_staleness_days", None)
    df_aligned = align_to_calendar(df_stale, master_calendar, max_staleness_days=max_staleness_cfg)
    logger.info("  - Aligned to business days")

    # Trim each series to its own first valid observation (ragged starts).
    # FRED market-data series begin in 2000; ALFRED economic series from ~2011.
    df_processed = trim_to_own_start(df_aligned)

    _log_summary("Aligned", df_processed, time.time() - stage_start)

    # =================================================================
    # STAGE 4: VALIDATE
    # =================================================================
    _log_stage_header(4, 5, "Validate")
    stage_start = time.time()

    report = validate_data(df_processed, cfg=macro_cfg, expected_calendar=master_calendar)
    logger.info("")
    print_validation_report(report)

    validation_time = time.time() - stage_start
    logger.info("  - Validation completed in %.2fs", validation_time)

    # Check if validation passed
    if report["status"] != "PASS":
        logger.warning("Validation FAILED — %d critical issue(s):", len(report["issues"]))
        for issue in report["issues"]:
            logger.warning("  [ISSUE] %s", issue)
    if report["warnings"]:
        logger.warning("Validation warnings (%d):", len(report["warnings"]))
        for warning in report["warnings"]:
            logger.warning("  [WARN] %s", warning)

    # =================================================================
    # STAGE 5: SAVE
    # =================================================================
    _log_stage_header(5, 5, "Save")
    stage_start = time.time()

    output_path = Path(macro_cfg["processed_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_processed.to_parquet(output_path, index=False)

    file_size = output_path.stat().st_size / (1024 * 1024)  # Convert to MB
    save_time = time.time() - stage_start

    logger.info("  - Saved to: %s", output_path)
    logger.info("  - File size: %.2f MB", file_size)
    logger.info("  - Save completed in %.2fs", save_time)

    # =================================================================
    # PIPELINE SUMMARY
    # =================================================================
    total_time = time.time() - pipeline_start

    logger.info("=" * 100)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 100)
    logger.info("Total time: %.2fs", total_time)
    logger.info("Output: %s", output_path)
    logger.info("Status: %s", "PASS" if report["status"] == "PASS" else "FAIL")
    logger.info("=" * 100)

    return df_processed


if __name__ == "__main__":
    run_macro_data_pipeline()
