"""
Test script to demonstrate the transform parser functionality.
"""

import yaml
from pathlib import Path
from regime_ml.features.common.transform_parser import TransformParser

def main():
    # Load the YAML configuration
    config_path = Path(__file__).parent.parent / "configs" / "data" / "regime_universe.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create parser
    parser = TransformParser()
    
    # Parse transforms
    regime_config = config['regime_universe']
    transform_chains = parser.parse_yaml_config(regime_config)
    
    # Display results
    print("=" * 80)
    print("TRANSFORM PARSER RESULTS")
    print("=" * 80)
    
    for ticker_name, chains in transform_chains.items():
        print(f"\n{ticker_name.upper()}")
        print("-" * 40)
        
        for i, chain in enumerate(chains, 1):
            print(f"  Chain {i}: {chain}")
            
            # Show individual transforms
            for j, transform in enumerate(chain.transforms, 1):
                print(f"    Step {j}: {transform}")
        
        # Show generated feature names
        feature_names = parser.get_feature_names(ticker_name, chains)
        print(f"\n  Generated feature names:")
        for fname in feature_names:
            print(f"    - {fname}")
    
    print("\n" + "=" * 80)
    print(f"Total tickers: {len(transform_chains)}")
    total_features = sum(len(chains) for chains in transform_chains.values())
    print(f"Total features: {total_features}")
    print("=" * 80)


if __name__ == "__main__":
    main()
