# src/regime_ml/features/common/transforms/temporal.py

from .base import BaseTransform
import pandas as pd
import numpy as np


class Diff(BaseTransform):
    """
    Difference transform: x(t) - x(t-periods)
    """
    
    def _validate_params(self) -> None:
        if 'periods' not in self.params:
            raise ValueError("Diff requires 'periods' parameter")
    
    def _compute(self, series: pd.Series) -> pd.Series:
        periods = self.params['periods']
        return series.diff(periods=periods)


class PctChange(BaseTransform):
    """
    Percentage change: (x(t) - x(t-periods)) / x(t-periods)
    """
    
    def _validate_params(self) -> None:
        self.params.setdefault('periods', 1)
    
    def _compute(self, series: pd.Series) -> pd.Series:
        periods = self.params['periods']
        return series.pct_change(periods=periods)


class YoY(BaseTransform):
    """
    Year-over-year change.
    
    For business days: 252 periods
    Automatically adjusts for native frequency if provided.
    """
    
    def _validate_params(self) -> None:
        # Default to 252 business days
        self.params.setdefault('periods', 252)
        self.params.setdefault('method', 'pct_change')  # or 'diff'
    
    def _compute(self, series: pd.Series) -> pd.Series:
        periods = self.params['periods']
        method = self.params['method']
        
        if method == 'pct_change':
            return series.pct_change(periods=periods)
        elif method == 'diff':
            return series.diff(periods=periods)
        else:
            raise ValueError(f"Unknown method: {method}")


class Returns(BaseTransform):
    """
    Returns calculation: log or simple.
    """
    
    def _validate_params(self) -> None:
        self.params.setdefault('method', 'log')  # 'log' or 'simple'
        self.params.setdefault('periods', 1)
    
    def _compute(self, series: pd.Series) -> pd.Series:
        periods = self.params['periods']
        method = self.params['method']
        
        if method == 'log':
            return (series / series.shift(periods)).apply(np.log)
        elif method == 'simple':
            return series.pct_change(periods=periods)
        else:
            raise ValueError(f"Unknown method: {method}")


class Level(BaseTransform):
    """
    Identity transform (no-op).
    Useful for consistent API when you want raw values.
    """
    
    def _validate_params(self) -> None:
        pass
    
    def _compute(self, series: pd.Series) -> pd.Series:
        return series.copy()