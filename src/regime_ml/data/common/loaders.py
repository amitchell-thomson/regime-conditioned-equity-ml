"""Common data loading utilities for all data types."""

from pathlib import Path
from typing import Union
import pandas as pd


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path: The project root directory path.
    """
    # Assumes this file is in src/regime_ml/data/common/
    return Path(__file__).parent.parent.parent.parent.parent


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
