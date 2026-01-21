"""Macro data pipeline for complete data processing workflow."""

import time
from pathlib import Path
import pandas as pd

from regime_ml.utils import load_configs
from regime_ml.data.common import create_master_calendar
from regime_ml.data.macro import (
    load_raw_data,
    select_data,
    clean_data,
    add_staleness_indicators,
    align_to_calendar,
    trim_to_common_start,
    validate_data,
    print_validation_report,
)


def _print_stage_header(stage_num: int, total_stages: int, title: str) -> None:
    """Print a consistent stage header."""
    print(f"\n[{stage_num}/{total_stages}] {title}")
    print("-" * 100)


def _print_summary(label: str, df: pd.DataFrame, timing: float | None = None) -> None:
    """Print concise summary statistics."""
    rows = len(df)
    series = df['series_code'].nunique()
    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    
    timing_str = f" ({timing:.2f}s)" if timing is not None else ""
    print(f"  ✓ {label}: {rows:,} rows, {series} series, {date_range}{timing_str}")


def run_macro_pipeline() -> pd.DataFrame:
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
    print("\n" + "=" * 100)
    print("MACRO DATA PIPELINE")
    print("=" * 100)
    
    # Load configuration
    cfg = load_configs()
    macro_cfg = cfg["macro_data"]["regime_universe"]
    
    # =================================================================
    # STAGE 1: LOAD
    # =================================================================
    _print_stage_header(1, 5, "Load")
    stage_start = time.time()
    
    df_raw = load_raw_data(macro_cfg["data_path"], macro_cfg["raw_path"])
    
    _print_summary("Loaded", df_raw, time.time() - stage_start)
    
    # =================================================================
    # STAGE 2: SELECT & CLEAN
    # =================================================================
    _print_stage_header(2, 5, "Select & Clean")
    stage_start = time.time()
    
    df_selected = select_data(df_raw, macro_cfg)
    print(f"  → Selected {df_selected['series_code'].nunique()} series from config")
    
    df_clean = clean_data(df_selected)
    print(f"  → Cleaned data (types, weekends, deduplication)")
    
    _print_summary("Processed", df_clean, time.time() - stage_start)
    
    # =================================================================
    # STAGE 3: ALIGN
    # =================================================================
    _print_stage_header(3, 5, "Align to Calendar")
    stage_start = time.time()
    
    # Create master business day calendar
    master_calendar = create_master_calendar(
        macro_cfg["lookback"]["start"],
        macro_cfg["lookback"]["end"]
    )
    print(f"  → Created calendar: {len(master_calendar):,} business days")
    
    # Add staleness tracking
    df_stale = add_staleness_indicators(df_clean, master_calendar)
    print(f"  → Added staleness indicators")
    
    # Align to calendar with forward-fill
    df_aligned = align_to_calendar(df_stale, master_calendar)
    print(f"  → Aligned to business days")
    
    # Trim to common start date (remove leading nulls)
    df_processed = trim_to_common_start(df_aligned)
    
    _print_summary("Aligned", df_processed, time.time() - stage_start)
    
    # =================================================================
    # STAGE 4: VALIDATE
    # =================================================================
    _print_stage_header(4, 5, "Validate")
    stage_start = time.time()
    
    report = validate_data(
        df_processed,
        cfg=macro_cfg,
        expected_calendar=master_calendar
    )
    print("\n")
    print_validation_report(report)
    
    validation_time = time.time() - stage_start
    print(f"  ✓ Validation completed in {validation_time:.2f}s")
    
    # Check if validation passed
    if report["status"] != "PASS":
        print(f"\n⚠️  WARNING: Validation found {len(report['issues'])} critical issues!")
    
    # =================================================================
    # STAGE 5: SAVE
    # =================================================================
    _print_stage_header(5, 5, "Save")
    stage_start = time.time()
    
    output_path = Path(macro_cfg["processed_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_processed.to_parquet(output_path, index=False)
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # Convert to MB
    save_time = time.time() - stage_start
    
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ File size: {file_size:.2f} MB")
    print(f"  ✓ Save completed in {save_time:.2f}s")
    
    # =================================================================
    # PIPELINE SUMMARY
    # =================================================================
    total_time = time.time() - pipeline_start
    
    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)
    print(f"Total time: {total_time:.2f}s")
    print(f"Output: {output_path}")
    print(f"Status: {'✓ PASS' if report['status'] == 'PASS' else '✗ FAIL'}")
    print("=" * 100 + "\n")
    
    return df_processed


if __name__ == "__main__":
    run_macro_pipeline()
