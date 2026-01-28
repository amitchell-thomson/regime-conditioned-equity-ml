"""
Macro Feature Pipeline

This module generates ML features from processed macro indicators by applying
transform chains defined in the regime_universe YAML configuration.

Key behaviors:
- Uses staleness_mode='strict' by default: computes transforms only on actual
  data points (respecting is_new_data flags), then forward-fills to full index
- For monthly indicators forward-filled to daily frequency, this means:
    - Transforms compute on ~312 actual monthly observations (not 6500 daily rows)
    - Results are forward-filled to daily frequency for alignment
    - Statistical operations (e.g., ZScore(window=60)) use 60 actual monthly points
- All features are automatically aligned to a common date index via pandas
"""

import pandas as pd
import yaml

from regime_ml.features.common.transform_parser import TransformParser
from regime_ml.features.macro.validator import validate_macro_features
from regime_ml.utils.config import load_configs
from regime_ml.data.common.loaders import load_dataframe
from regime_ml.features.macro.selection import get_feature_groups, get_top_features


def create_feature_metadata(features: pd.DataFrame, frequency_map: dict) -> None:
    feature_metadata = []
    for col in features.columns:
        # Parse feature name
        parts = col.split('_')
        indicator = parts[0]
        transforms = '_'.join(parts[1:])
        
        feature_metadata.append({
            'name': col,
            'indicator': indicator,
            'transforms': transforms,
            'frequency': frequency_map.get(indicator, 'unknown'),
            'mean': float(features[col].mean()),
            'std': float(features[col].std()),
            'missing_pct': float(features[col].isna().sum() / len(features) * 100),
            'n_outliers_5sigma': int((features[col].abs() > 5).sum())
        })

    # Save as YAML or JSON
    with open('data/features/feature_metadata.yaml', 'w') as f:
        yaml.dump({'features': feature_metadata, 'n_features': len(feature_metadata)}, f)

    print(f"Created metadata for {len(feature_metadata)} features")


def run_macro_feature_pipeline() -> pd.DataFrame:
    """
    Generate macro features from processed data using YAML-defined transform chains.
    
    Pipeline steps:
    1. Load processed macro data (forward-filled to daily business day frequency)
    2. Parse transform chains from YAML configuration
    3. For each indicator:
            - Extract series and staleness flags (is_new_data)
            - Apply each transform chain with staleness_mode='strict':
                - Computes only on actual data points (is_new_data=True)
                - Forward-fills results to full daily index
            - Generate descriptive feature names
    4. Save features to parquet with all columns aligned to common index
    
    Returns:
        DataFrame with all generated features, indexed by date
        
    Output:
        Saves features to path specified in regime_universe.features_path
        
    Notes:
        - First ~500-700 days will have nulls due to window burn-in periods
        - Monthly indicators show step functions (constant within month)
        - Daily indicators show daily variation
        - Complete coverage typically achieved by ~2006-2007
    """
    # Load configuration
    cfg = load_configs()
    regime_cfg = cfg["macro_data"]["regime_universe"]
    processed_path = regime_cfg["processed_path"]
    
    # Load processed macro data
    # Expected format: columns=['date', 'series_code', 'value', 'is_new_data']
    processed_data = load_dataframe(processed_path)
    
    # Parse transform chains from YAML configuration
    # Returns: {series_code: [ChainedTransform, ...]}
    transform_parser = TransformParser()
    transform_chains = transform_parser.parse_yaml_config(regime_cfg)
    
    # Extract frequency map from YAML configuration
    # Maps series_code -> frequency ('daily', 'weekly', 'monthly')
    frequency_map = {}
    for ticker_name, ticker_config in regime_cfg.get('series', {}).items():
        series_code = ticker_config.get('id')
        frequency = ticker_config.get('frequency', 'daily')  # Default to daily if not specified
        if series_code:
            frequency_map[series_code] = frequency
    
    # Initialize feature dataframe
    # Pandas will auto-align features to common index when assigning columns
    feature_data = pd.DataFrame()
    
    # Process each macro indicator
    for series_code in processed_data["series_code"].unique():
        # Extract data for this indicator
        ticker_df = processed_data[processed_data["series_code"] == series_code].set_index("date")
        
        # Staleness flags: True = actual data point, False = forward-filled
        is_new_data = pd.Series(ticker_df["is_new_data"])
        
        # Indicator values (already forward-filled to daily frequency)
        series = pd.Series(ticker_df["value"])
        
        # Get transform chains for this indicator
        chains = transform_chains[series_code]
        
        # Generate descriptive feature names
        # Format: {SERIES_CODE}_{transform1}_{param1}_{transform2}_{param2}...
        feature_names = transform_parser.get_feature_names(series_code, chains)
        
        # Apply each transform chain
        for feature_name, chain in zip(feature_names, chains):
            # Transform with staleness awareness:
            # - Computes on actual data points only (where is_new_data=True)
            # - Forward-fills results to full daily index
            # - staleness_mode='strict' is the default
            transformed_series = chain.transform(
                series=series,
                is_new_data=is_new_data
            )
            
            # Add feature to dataframe (auto-aligns by date index)
            feature_data[feature_name] = transformed_series
    
    # Validate features before saving
    validation_results = validate_macro_features(feature_data, frequency_map)
    
    # Check if validation passed
    errors = [r for r in validation_results if r.severity == 'error' and not r.passed]
    if errors:
        print("\n[WARN] Validation found errors but continuing with save.")
        print("       Review validation results above before using these features.")
    
    # Save raw_features to parquet
    raw_output_path = regime_cfg["raw_features_path"]
    feature_data.to_parquet(raw_output_path)

    # Drop burn-in period
    feature_data_ready = feature_data.dropna()

    # Get top features
    top_features = get_top_features(n=8)
    feature_data_ready = feature_data_ready[top_features]

    # Save selected_features to parquet
    selected_output_path = regime_cfg["ready_features_path"]
    feature_data_ready.to_parquet(selected_output_path)
    
    print(f"\nSelected {len(feature_data_ready.columns)} features")
    print(f"Date range: {feature_data_ready.index.min()} to {feature_data_ready.index.max()}")
    print(f"Number of rows: {len(feature_data_ready)}")
    print(f"Saved to: {raw_output_path} and {selected_output_path}")

    # Create feature metadata
    create_feature_metadata(feature_data_ready, frequency_map) # type: ignore

    return feature_data


if __name__ == "__main__":
    run_macro_feature_pipeline()
