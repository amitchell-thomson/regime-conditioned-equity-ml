"""Utility functions for loading and parsing configuration files."""

from pathlib import Path
from typing import Dict, Any, Union
import yaml


def load_configs() -> Dict[str, Any]:
    """Load all configuration files into a single dictionary.
    
    Returns:
        Dict[str, Any]: Configuration dictionary with keys:
            - 'data': Data configuration from regime_universe.yaml
            - 'regimes': Regime configuration from regime_definitions.yaml
            - 'models': Model configuration from model_definitions.yaml
    
    Example:
        >>> cfg = load_configs()
        >>> print(cfg['data'].keys())
        >>> print(cfg['regimes'].keys())
    """
    # Get project root (assumes this file is in src/regime_ml/utils/)
    project_root = Path(__file__).parent.parent.parent.parent
    
    config_paths = {
        "data": project_root / "configs/data/regime_universe.yaml",
        "regimes": project_root / "configs/regimes/regime_config.yaml",
        "models": project_root / "configs/models/model_config.yaml",
    }
    
    cfg = {}
    for key, path in config_paths.items():
        if path.exists():
            with open(path, "r") as f:
                cfg[key] = yaml.safe_load(f)
        else:
            print(f"Warning: Config file not found: {path}")
            cfg[key] = {}
    
    return cfg