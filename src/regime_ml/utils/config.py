"""Utility functions for loading and parsing configuration files."""

import logging
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_configs() -> Dict[str, Any]:
    """Load all configuration files into a single dictionary.

    Reads YAML configs and applies environment variable overrides:
      - MACRO_DATA_PATH overrides macro_data.regime_universe.data_path

    Set overrides in a .env file at the project root (see .env.example).

    Returns:
        Dict[str, Any]: Configuration dictionary with keys:
            - 'macro_data': Data configuration from regime_universe.yaml
            - 'equity_data': Data configuration from equities_universe.yaml
            - 'regimes': Regime configuration from regime_config.yaml
            - 'models': Model configuration from model_config.yaml

    Example:
        >>> cfg = load_configs()
        >>> print(cfg['macro_data'].keys())
        >>> print(cfg['regimes'].keys())
    """
    load_dotenv()

    # Get project root (assumes this file is in src/regime_ml/utils/)
    project_root = Path(__file__).parent.parent.parent.parent

    config_paths = {
        "macro_data": project_root / "configs/data/regime_universe.yaml",
        "equity_data": project_root / "configs/data/equities_universe.yaml",
        "regimes": project_root / "configs/regimes/regime_config.yaml",
        "models": project_root / "configs/models/model_config.yaml",
    }

    cfg: Dict[str, Any] = {}
    for key, path in config_paths.items():
        if path.exists():
            with open(path, "r") as f:
                cfg[key] = yaml.safe_load(f)
        else:
            logger.warning("Config file not found: %s", path)
            cfg[key] = {}

    # Apply environment variable overrides
    env_data_path = os.environ.get("MACRO_DATA_PATH")
    if env_data_path:
        cfg.setdefault("macro_data", {}).setdefault("regime_universe", {})[
            "data_path"
        ] = env_data_path

    return cfg
