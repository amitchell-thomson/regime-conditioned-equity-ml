from tkinter import W
from .base import BaseTransform
import pandas as pd
import numpy as np

class ZScore(BaseTransform):
    """
    Rolling z-score normalization.

    Formula: (x - mean(x, window)) / std(x, window)
    """
    def _validate_params(self) -> None:
        if "window" not in self.params:
            raise ValueError("Window parameter is required for ZScore transform")

    def _compute(self, series: pd.Series) -> pd.Series:
        window = self.params["window"]

        rolling = series.rolling(window)
        mean = rolling.mean()
        std = rolling.std()

        z_score = (series - mean) / std.replace(0, np.nan)

        return z_score

class MovingAverage(BaseTransform):
    """
    Rolling moving average.

    Formula: mean(x, window)
    """
    def _validate_params(self) -> None:
        if "window" not in self.params:
            raise ValueError("Window parameter is required for MovingAverage transform")

    def _compute(self, series: pd.Series) -> pd.Series:
        window = self.params["window"]

        moving_average = series.rolling(window).mean()

        return moving_average # type: ignore

class ExponentialMovingAverage(BaseTransform):
    """
    Exponential moving average.

    Formula: EMA(x, window) = alpha * x + (1 - alpha) * EMA(x, window-1)
    """
    def _validate_params(self) -> None:
        if "span" not in self.params and "halflife" not in self.params:
            raise ValueError("EMA requires 'span' or 'halflife' parameter")

    def _compute(self, series: pd.Series) -> pd.Series:
        span = self.params.get("span")
        halflife = self.params.get("halflife")
        if span:
            return series.ewm(span=span).mean() # type: ignore
        else:
            return series.ewm(halflife=halflife).mean() # type: ignore

class RollingStd(BaseTransform):
    """
    Rolling standard deviation.

    Formula: std(x, window)
    """
    def _validate_params(self) -> None:
        if "window" not in self.params:
            raise ValueError("Window parameter is required for RollingStd transform")

    def _compute(self, series: pd.Series) -> pd.Series:
        window = self.params["window"]

        rolling = series.rolling(window)
        std = rolling.std()
        
        return std # type: ignore
            