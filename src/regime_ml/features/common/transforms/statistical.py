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

        return moving_average  # type: ignore[return-value]  # pandas rolling().mean() returns Series[Any] — mypy cannot narrow from input type


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
            return series.ewm(span=span).mean()  # type: ignore[return-value]  # pandas ewm().mean() returns Series[Any] — mypy cannot narrow
        else:
            return series.ewm(halflife=halflife).mean()  # type: ignore[return-value]  # pandas ewm().mean() returns Series[Any] — mypy cannot narrow


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

        return std  # type: ignore[return-value]  # pandas rolling().std() returns Series[Any] — mypy cannot narrow from input type


class Winsorize(BaseTransform):
    """
    Clip a z-scored series to the symmetric interval [-sigma, +sigma].

    Designed to bound extreme tail observations (e.g. VIX spikes, NFCI stress)
    before HMM fitting. Assumes the input is already standardised (z-scored).

    Parameters:
        sigma (float): Symmetric clip bound. Must be positive.
                       Set in regime_universe.yaml — never hardcode this value.

    Staleness behaviour: pointwise clip — correct under both 'strict' and
    'ignore' staleness modes. Clipping a forward-filled copy of a clipped
    value is idempotent, so the base class staleness logic handles this
    without any special treatment.
    """

    def _validate_params(self) -> None:
        if "sigma" not in self.params:
            raise ValueError(
                "Winsorize requires a 'sigma' parameter (symmetric clip bound). "
                "Set this in regime_universe.yaml."
            )
        if self.params["sigma"] <= 0:
            raise ValueError(
                f"Winsorize sigma must be positive, got {self.params['sigma']}."
            )

    def _compute(self, series: pd.Series) -> pd.Series:
        sigma = self.params["sigma"]
        return series.clip(lower=-sigma, upper=sigma)
