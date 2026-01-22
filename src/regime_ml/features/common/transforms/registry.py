from typing import Dict, Type
from .base import BaseTransform
from .statistical import ZScore, MovingAverage, ExponentialMovingAverage, RollingStd
from .temporal import Diff, PctChange, YoY, Returns, Level


class TransformRegistry:
    """
    Registry for all available transforms.
    Maps string names to transform classes.
    """
    
    _transforms: Dict[str, Type[BaseTransform]] = {
        # Statistical
        'z_score': ZScore,
        'zscore': ZScore,
        'ma': MovingAverage,
        'moving_average': MovingAverage,
        'sma': MovingAverage,
        'ema': ExponentialMovingAverage,
        'ewm': ExponentialMovingAverage,
        'std': RollingStd,
        'rolling_std': RollingStd,
        
        # Temporal
        'diff': Diff,
        'difference': Diff,
        'pct_change': PctChange,
        'pct': PctChange,
        'returns': Returns,
        'yoy': YoY,
        'level': Level,
    }
    
    @classmethod
    def get(cls, name: str) -> Type[BaseTransform]:
        """Get transform class by name."""
        name_lower = name.lower().replace('-', '_')
        
        if name_lower not in cls._transforms:
            available = ', '.join(cls._transforms.keys())
            raise ValueError(
                f"Unknown transform: '{name}'. "
                f"Available transforms: {available}"
            )
        
        return cls._transforms[name_lower]
    
    @classmethod
    def register(cls, name: str, transform_class: Type[BaseTransform]) -> None:
        """Register a new transform."""
        cls._transforms[name.lower()] = transform_class
    
    @classmethod
    def list_transforms(cls) -> list[str]:
        """List all available transforms."""
        return sorted(cls._transforms.keys())
    
    @classmethod
    def create(cls, name: str, **params) -> BaseTransform:
        """Create transform instance by name."""
        transform_class = cls.get(name)
        return transform_class(**params)