"""
Transform parser for YAML-based feature engineering configurations.

Parses compact structured transform notation into ChainedTransform objects.
Example:
    [{diff: {periods: 21}}, {z_score: {window: 252}}]
    -> ChainedTransform([Diff(periods=21), ZScore(window=252)])
"""

from typing import Dict, List, Any, Union
from .transforms import TransformRegistry, ChainedTransform, BaseTransform


class TransformParser:
    """
    Parse YAML transform specifications into ChainedTransform objects.
    
    Supports compact structured format:
        - [level, {z_score: {window: 63}}]
        - [{diff: {periods: 21}}, {z_score: {window: 252}}]
        - [{ma: {window: 21}}, {diff: {periods: 21}}, {z_score: {window: 252}}]
    """
    
    def __init__(self):
        self.registry = TransformRegistry
    
    def parse_yaml_config(self, yaml_config: Dict[str, Any]) -> Dict[str, List[ChainedTransform]]:
        """
        Parse entire YAML configuration and return dictionary of ticker -> transform chains.
        
        Args:
            yaml_config: Dict from loaded YAML file
            
        Returns:
            Dict mapping ticker names to lists of ChainedTransform objects
            
        Example:
            {
                'vix': [
                    ChainedTransform([Level(), ZScore(window=63)]),
                    ChainedTransform([Diff(periods=5), ZScore(window=126)])
                ],
                'ust10y': [...]
            }
        """
        result = {}
        
        series = yaml_config.get('series', {})
        
        for ticker_name, ticker_config in series.items():
            transform_specs = ticker_config.get('transforms', [])
            
            # Parse each transform chain specification
            transform_chains = [
                self.parse_chain(chain_spec) 
                for chain_spec in transform_specs
            ]
            
            result[ticker_name] = transform_chains
        
        return result
    
    def parse_chain(self, chain_spec: List[Union[str, Dict]]) -> ChainedTransform:
        """
        Parse a single transform chain specification.
        
        Args:
            chain_spec: List representing a transform chain
                Examples:
                    [level, {z_score: {window: 63}}]
                    [{diff: {periods: 21}}, {z_score: {window: 252}}]
                    
        Returns:
            ChainedTransform object with all transforms in sequence
        """
        transforms = []
        
        for step in chain_spec:
            transform_obj = self._parse_transform_step(step)
            transforms.append(transform_obj)
        
        return ChainedTransform(transforms)
    
    def _parse_transform_step(self, step: Union[str, Dict]) -> BaseTransform:
        """
        Parse a single transform step.
        
        Args:
            step: Either a string (transform name) or dict {transform_name: {params}}
            
        Returns:
            BaseTransform instance
            
        Examples:
            'level' -> Level()
            {z_score: {window: 63}} -> ZScore(window=63)
            {diff: {periods: 21}} -> Diff(periods=21)
        """
        if isinstance(step, str):
            # Simple transform with no parameters: 'level'
            return self.registry.create(step)
        
        elif isinstance(step, dict):
            # Transform with parameters: {z_score: {window: 63}}
            if len(step) != 1:
                raise ValueError(f"Transform step must have exactly one key, got: {step}")
            
            transform_name = list(step.keys())[0]
            params = step[transform_name]
            
            if params is None:
                params = {}
            elif not isinstance(params, dict):
                raise ValueError(f"Transform parameters must be a dict, got: {params}")
            
            return self.registry.create(transform_name, **params)
        
        else:
            raise ValueError(f"Invalid transform step type: {type(step)}")
    
    def get_feature_names(
        self, 
        ticker_name: str, 
        transform_chains: List[ChainedTransform]
    ) -> List[str]:
        """
        Generate feature names for a ticker's transform chains.
        
        Args:
            ticker_name: Base ticker name (e.g., 'vix', 'ust10y')
            transform_chains: List of ChainedTransform objects
            
        Returns:
            List of feature names
            
        Example:
            ticker_name='vix', chains=[ChainedTransform([Level(), ZScore(window=63)])]
            -> ['vix_level_zscore_63']
        """
        feature_names = []
        
        for i, chain in enumerate(transform_chains):
            # Build feature name from transform chain
            chain_parts = []
            
            for transform in chain.transforms:
                # Get transform name
                name = transform.__class__.__name__.lower()
                
                # Add key parameters to name
                if hasattr(transform, 'params') and transform.params:
                    # Extract key params (window, periods, etc.)
                    key_params = []
                    for param_name in ['window', 'periods', 'span', 'halflife']:
                        if param_name in transform.params:
                            key_params.append(str(transform.params[param_name]))
                    
                    if key_params:
                        name = f"{name}_{'_'.join(key_params)}"
                
                chain_parts.append(name)
            
            feature_name = f"{ticker_name}_{'_'.join(chain_parts)}"
            feature_names.append(feature_name)
        
        return feature_names


# Convenience function for easy import
def parse_transforms(yaml_config: Dict[str, Any]) -> Dict[str, List[ChainedTransform]]:
    """
    Parse YAML configuration into transform chains.
    
    Args:
        yaml_config: Dictionary from loaded YAML file
        
    Returns:
        Dict mapping ticker names to lists of ChainedTransform objects
    """
    parser = TransformParser()
    return parser.parse_yaml_config(yaml_config)
