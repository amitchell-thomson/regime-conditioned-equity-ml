from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

class BaseTransform(ABC):
    """
    Base class for all feature transforms.
    
    Design principles:
    1. Transforms operate on Series, not DataFrames (single column in, single column out)
    2. Support for staleness-aware computation
    3. Chainable (output of one transform can be input to another)
    4. Immutable configuration
    """

    def __init__(self, **params):
        """
        Initialize transform with parameters.
        
        Common parameters:
            window: int - lookback window for rolling operations
            min_periods: int - minimum observations required
            staleness_mode: str - 'strict' (actual data only) or 'filled' (use forward-filled)
        """
        self.params = params
        self._validate_params()

    @abstractmethod
    def _validate_params(self) -> None:
        """
        Validate transform parameters. Subclasses implement this.
        """
        pass

    @abstractmethod
    def _compute(self, series: pd.Series) -> pd.Series:
        """
        Core transform logic. Subclasses implement this.
        """
        pass

    def transform(
        self,
        series: pd.Series,
        is_new_data: Optional[pd.Series] = None,
        staleness_mode: str = "strict"
    ) -> pd.Series:
        """
        Apply transform to a series.
        
        Args:
            series: Input series with DatetimeIndex
            is_new_data: Boolean series indicating actual vs forward-filled data
            staleness_mode: 
                - 'strict': Compute only on actual data points, then forward-fill
                - 'ignore': Ignore staleness and compute on all data (could be computing of forward-filled data)
                - 'weighted': Weight by data freshness
        
        Returns:
            Transformed series with same index as input
        """
        if staleness_mode == "strict" and is_new_data is not None:
            # compute on actual data points
            actual_data = series[is_new_data == True]
            result = self._compute(actual_data) # type: ignore

            # forward-fill to original index
            result = result.reindex(series.index).ffill()

        elif staleness_mode == "ignore":
            # compute on forward-filled data
            result = self._compute(series)

        elif staleness_mode == "weighted":
            print("Have not implemented weighted staleness mode yet")
            result = self._compute(series)

        else:
            raise ValueError(f"Invalid staleness mode: {staleness_mode}")

        return result

    def __call__(self, series: pd.Series, **kwargs) -> pd.Series:
        """Allow transform to be called as a function."""
        return self.transform(series, **kwargs)

    def __chain__(self, other_transform: "BaseTransform") -> "ChainedTransform":
        """Chain this transform with another"""
        return ChainedTransform([self, other_transform])

    def __repr__(self) -> str:
        params_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        return f"{self.__class__.__name__}({params_str})"


class ChainedTransform(BaseTransform):
    """
    Chain multiple transforms together.
    
    Example:
        transform = Diff(periods=21).chain(ZScore(window=252))
        # Equivalent to: z_score(diff(x, 21), 252)
    """

    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms
        self.params = {"transforms": transforms}

    def _validate_params(self) -> None:
        if not self.transforms:
            raise ValueError("ChainedTransform must have at least one transform")
    
    def _compute(self, series: pd.Series) -> pd.Series:
        result = series
        for transform in self.transforms:
            result = transform._compute(result)
        return result
    
    def transform(
        self,
        series: pd.Series,
        is_new_data: Optional[pd.Series] = None,
        staleness_mode: str = "strict"
    ) -> pd.Series:
        """Apply transforms sequentially."""
        result = series
        for transform in self.transforms:
            result = transform.transform(result, is_new_data, staleness_mode)
        return result
