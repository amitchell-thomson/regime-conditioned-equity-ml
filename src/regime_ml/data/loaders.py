"""Data loading utilities for loading dataframes from various locations."""

from pathlib import Path
from typing import Union, Optional, List
import pandas as pd
from tqdm import tqdm


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path: The project root directory path.
    """
    # Assumes this file is in src/regime_ml/data/
    return Path(__file__).parent.parent.parent.parent


def load_dataframe(
    filepath: Union[str, Path],
    relative_to_root: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Load a dataframe from anywhere in the project.
    
    This function automatically handles:
    - Relative paths (resolved from project root by default)
    - Absolute paths
    - Multiple file formats (parquet, csv, excel, feather, json, pickle)
    
    Args:
        filepath: Path to the file to load. Can be relative or absolute.
        relative_to_root: If True, relative paths are resolved from project root.
                         If False, relative paths are resolved from current working directory.
        **kwargs: Additional keyword arguments passed to the pandas read function.
    
    Returns:
        pd.DataFrame: The loaded dataframe.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported.
    
    Examples:
        >>> # Load from project root
        >>> df = load_dataframe("data/raw/regime_raw.parquet")
        
        >>> # Load with absolute path
        >>> df = load_dataframe("/absolute/path/to/data.csv")
        
        >>> # Load relative to current directory
        >>> df = load_dataframe("../../data/file.csv", relative_to_root=False)
        
        >>> # Pass additional arguments to pandas reader
        >>> df = load_dataframe("data/file.csv", index_col=0, parse_dates=True)
    """
    filepath = Path(filepath)
    
    # Resolve the path
    if not filepath.is_absolute():
        if relative_to_root:
            filepath = get_project_root() / filepath
        # else: relative to current working directory (default Path behavior)
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            f"Project root: {get_project_root()}\n"
            f"Current directory: {Path.cwd()}"
        )
    
    # Determine file format and load accordingly
    suffix = filepath.suffix.lower()
    
    loaders = {
        '.parquet': pd.read_parquet,
        '.pq': pd.read_parquet,
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.feather': pd.read_feather,
        '.ftr': pd.read_feather,
        '.json': pd.read_json,
        '.pkl': pd.read_pickle,
        '.pickle': pd.read_pickle,
    }
    
    if suffix not in loaders:
        raise ValueError(
            f"Unsupported file format: {suffix}\n"
            f"Supported formats: {', '.join(loaders.keys())}"
        )
    
    # Load the dataframe
    loader_func = loaders[suffix]
    return loader_func(filepath, **kwargs)


def load_raw_data(
    source_dir: Union[str, Path] = "/Users/alecmitchell-thomson/Desktop/Coding/quant-data/macro",
    output_path: Union[str, Path] = "data/raw/macro_raw.parquet",
    relative_output: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """Load raw data from a source directory and combine into a single dataframe.
    
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
                    Default is "data/raw/regime_raw.parquet"
        relative_output: If True, output_path is relative to project root.
                        If False, output_path is treated as absolute.
        verbose: If True, print progress information.
    
    Returns:
        pd.DataFrame: The combined dataframe with all raw data.
    
    Raises:
        FileNotFoundError: If source_dir doesn't exist.
        ValueError: If no parquet files are found in source_dir.
    
    Examples:
        >>> # Use defaults (load from quant-data, save to data/raw)
        >>> df = load_raw_data(
        ...     source_dir="/Users/alecmitchell-thomson/Desktop/Coding/quant-data/macro",
        ...     output_path="data/raw/regime_raw.parquet"
        ... )
        
        >>> # Custom source and output
        >>> df = load_raw_data(
        ...     source_dir="/path/to/data",
        ...     output_path="data/processed/macro.parquet"
        ... )
    """
    source_dir = Path(source_dir)
    
    # Check if source directory exists
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Find all parquet files recursively
    parquet_files = sorted(source_dir.rglob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {source_dir}")
    
    if verbose:
        print(f"Found {len(parquet_files)} parquet files in {source_dir}")
        print("Files to combine:")
        for f in parquet_files:
            print(f"  - {f.relative_to(source_dir)}")
    
    # Load and combine all parquet files
    dataframes: List[pd.DataFrame] = []
    
    iterator = tqdm(parquet_files, desc="Loading parquet files") if verbose else parquet_files
    
    for parquet_file in iterator:
        try:
            df = pd.read_parquet(parquet_file)
            dataframes.append(df)
            if verbose and not isinstance(iterator, tqdm):
                print(f"  Loaded: {parquet_file.name} ({len(df):,} rows)")
        except Exception as e:
            print(f"  Warning: Failed to load {parquet_file}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No dataframes were successfully loaded")
    
    # Combine all dataframes
    if verbose:
        print("\nCombining dataframes...")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Sort by date and series_code for consistency
    if 'date' in combined_df.columns and 'series_code' in combined_df.columns:
        combined_df = combined_df.sort_values(['series_code', 'date']).reset_index(drop=True)
    
    if verbose:
        print(f"Combined dataframe shape: {combined_df.shape}")
        print(f"Columns: {combined_df.columns.tolist()}")
        if 'category' in combined_df.columns:
            print(f"Categories: {combined_df['category'].unique().tolist()}")
        if 'series_code' in combined_df.columns:
            print(f"Unique series: {combined_df['series_code'].nunique()}")
    
    # Resolve output path
    output_path = Path(output_path)
    if not output_path.is_absolute() and relative_output:
        output_path = get_project_root() / output_path
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    if verbose:
        print(f"\nSaving to: {output_path}")
    
    combined_df.to_parquet(output_path, index=False)
    
    if verbose:
        print(f"✓ Successfully saved {len(combined_df):,} rows to {output_path}")
    
    return combined_df
