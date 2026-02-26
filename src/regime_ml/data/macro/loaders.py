"""Macro data loading utilities."""

import logging
from pathlib import Path
from typing import Union, List
import pandas as pd

from regime_ml.data.common import get_project_root

logger = logging.getLogger(__name__)


def load_raw_data(
    source_dir: Union[str, Path, None] = None,
    output_path: Union[str, Path] = "data/raw/macro_raw.parquet",
    relative_output: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load raw macro data from a source directory and combine into a single dataframe.

    This function recursively finds all .parquet files in the source directory,
    loads them, and concatenates them into a single dataframe. The combined
    dataframe is then saved to the specified output path.

    Expected data structure:
        - Each parquet file should have columns: series_code, date, value, series_name, category
        - Files are organized in subdirectories by category (rates, growth, inflation, etc.)

    Args:
        source_dir: Path to the directory containing raw data parquet files.
                   Default is the quant-data macro folder.
        output_path: Where to save the combined parquet file.
                    Default is "data/raw/macro_raw.parquet"
        relative_output: If True, output_path is relative to project root.
                        If False, output_path is treated as absolute.
        verbose: Deprecated. Use log level to control output verbosity.

    Returns:
        pd.DataFrame: The combined dataframe with all raw data.

    Raises:
        FileNotFoundError: If source_dir doesn't exist.
        ValueError: If no parquet files are found in source_dir.

    Examples:
        >>> # Use defaults (load from quant-data, save to data/raw)
        >>> df = load_raw_data(
        ...     source_dir="/Users/alecmitchell-thomson/Desktop/Coding/quant-data/macro",
        ...     output_path="data/raw/macro_raw.parquet"
        ... )

        >>> # Custom source and output
        >>> df = load_raw_data(
        ...     source_dir="/path/to/data",
        ...     output_path="data/processed/macro.parquet"
        ... )
    """
    if source_dir is None:
        raise ValueError(
            "source_dir is required. Set the MACRO_DATA_PATH environment variable "
            "(see .env.example) or pass source_dir explicitly. The pipeline reads "
            "this from configs/data/regime_universe.yaml via load_configs()."
        )

    source_dir = Path(source_dir)

    # Check if source directory exists
    if not source_dir.exists():
        raise FileNotFoundError("Source directory not found: %s" % source_dir)

    # Find all parquet files recursively, excluding ALFRED files
    parquet_files = sorted(
        f
        for f in source_dir.rglob("*.parquet")
        if not f.name.endswith("_alfred.parquet")
    )

    if not parquet_files:
        raise ValueError("No parquet files found in %s" % source_dir)

    logger.info("Found %d parquet files in %s", len(parquet_files), source_dir)
    for f in parquet_files:
        logger.debug("  - %s", f.relative_to(source_dir))

    # Load and combine all parquet files
    dataframes: List[pd.DataFrame] = []

    for i, parquet_file in enumerate(parquet_files, 1):
        logger.info("  [%d/%d] %s", i, len(parquet_files), parquet_file.name)
        try:
            df = pd.read_parquet(parquet_file)
            dataframes.append(df)
            logger.debug("  Loaded: %s (%d rows)", parquet_file.name, len(df))
        except Exception as e:
            logger.warning("Failed to load %s: %s", parquet_file, e)
            continue

    if not dataframes:
        raise ValueError("No dataframes were successfully loaded")

    logger.info("Combining dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Sort by date and series_code for consistency
    if "date" in combined_df.columns and "series_code" in combined_df.columns:
        combined_df = combined_df.sort_values(["series_code", "date"]).reset_index(
            drop=True
        )

    logger.info("Combined dataframe shape: %s", combined_df.shape)
    logger.debug("Columns: %s", combined_df.columns.tolist())
    if "category" in combined_df.columns:
        logger.debug("Categories: %s", combined_df["category"].unique().tolist())
    if "series_code" in combined_df.columns:
        logger.info("Unique series: %d", combined_df["series_code"].nunique())

    # Resolve output path
    output_path = Path(output_path)
    if not output_path.is_absolute() and relative_output:
        output_path = get_project_root() / output_path

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    logger.info("Saving to: %s", output_path)
    combined_df.to_parquet(output_path, index=False)
    logger.info("Successfully saved %d rows to %s", len(combined_df), output_path)

    return combined_df


def load_alfred_data(
    source_dir: Union[str, Path],
    output_path: Union[str, Path] = "data/raw/macro_alfred_raw.parquet",
    relative_output: bool = True,
) -> pd.DataFrame:
    """Load ALFRED vintage data from *_alfred.parquet files.

    ALFRED (Archival FRED) data tracks each value at its initial release date
    and subsequent revisions. This enables point-in-time reconstruction to
    eliminate publication-lag lookahead bias.

    ALFRED parquet files have columns:
        series_code, date (obs period), realtime_start (publication date),
        realtime_end, value, series_name, category

    This function recursively finds all files matching *_alfred.parquet in
    source_dir and concatenates them. Plain .parquet files (FRED) are excluded.

    Args:
        source_dir: Path to macro data directory (same as FRED data_path).
                   ALFRED files are identified by the _alfred.parquet suffix.
        output_path: Where to save combined raw ALFRED data.
        relative_output: If True, output_path is relative to project root.

    Returns:
        pd.DataFrame: Combined ALFRED data with columns:
            series_code, date, realtime_start, realtime_end, value,
            series_name, category

    Raises:
        FileNotFoundError: If source_dir does not exist.
        ValueError: If no ALFRED parquet files are found.
    """
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError("Source directory not found: %s" % source_dir)

    # Only load ALFRED files (identified by _alfred.parquet suffix)
    alfred_files = sorted(source_dir.rglob("*_alfred.parquet"))

    if not alfred_files:
        raise ValueError(
            "No ALFRED parquet files found in %s. "
            "Expected files matching '*_alfred.parquet'." % source_dir
        )

    logger.info("Found %d ALFRED parquet files in %s", len(alfred_files), source_dir)
    for f in alfred_files:
        logger.debug("  - %s", f.relative_to(source_dir))

    dataframes: List[pd.DataFrame] = []

    for i, alfred_file in enumerate(alfred_files, 1):
        logger.info("  [%d/%d] %s", i, len(alfred_files), alfred_file.name)
        try:
            df = pd.read_parquet(alfred_file)
            # Ensure date columns are parsed as timestamps
            for col in ("date", "realtime_start", "realtime_end"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            dataframes.append(df)
            logger.debug("  Loaded: %s (%d rows)", alfred_file.name, len(df))
        except Exception as e:
            logger.warning("Failed to load %s: %s", alfred_file, e)
            continue

    if not dataframes:
        raise ValueError("No ALFRED dataframes were successfully loaded")

    logger.info("Combining ALFRED dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)

    if "series_code" in combined_df.columns and "realtime_start" in combined_df.columns:
        combined_df = combined_df.sort_values(
            ["series_code", "realtime_start", "date"]
        ).reset_index(drop=True)

    logger.info(
        "Combined ALFRED shape: %s, %d series",
        combined_df.shape,
        combined_df["series_code"].nunique(),
    )

    # Resolve output path
    output_path = Path(output_path)
    if not output_path.is_absolute() and relative_output:
        output_path = get_project_root() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving ALFRED raw data to: %s", output_path)
    combined_df.to_parquet(output_path, index=False)
    logger.info("Successfully saved %d ALFRED rows", len(combined_df))

    return combined_df
