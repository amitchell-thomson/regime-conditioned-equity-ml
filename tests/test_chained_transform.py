"""Tests for ChainedTransform staleness guards in base.py."""

import numpy as np
import pandas as pd
import pytest

from regime_ml.features.common.transforms.base import ChainedTransform
from regime_ml.features.common.transforms.statistical import ZScore
from regime_ml.features.common.transforms.temporal import Diff


def _make_series(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.standard_normal(n), index=idx)


def test_chained_transform_compute_raises():
    """ChainedTransform._compute() must raise NotImplementedError to prevent staleness bypass."""
    chain = ChainedTransform([Diff(periods=5), ZScore(window=20)])
    series = _make_series()
    with pytest.raises(NotImplementedError, match="disabled"):
        chain._compute(series)


def test_chained_transform_transform_works():
    """ChainedTransform.transform() must still work correctly end-to-end."""
    chain = ChainedTransform([Diff(periods=5), ZScore(window=20)])
    series = _make_series(n=100)
    result = chain.transform(series, is_new_data=None, staleness_mode="ignore")
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)


def test_chained_transform_transform_strict_mode():
    """transform() in strict mode should compute only on is_new_data=True rows."""
    chain = ChainedTransform([Diff(periods=1), ZScore(window=10)])
    series = _make_series(n=60)
    # Mark every 5th day as real data
    is_new = pd.Series(False, index=series.index)
    is_new.iloc[::5] = True
    result = chain.transform(series, is_new_data=is_new, staleness_mode="strict")
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)


def test_weighted_staleness_mode_raises():
    """staleness_mode='weighted' is unimplemented and must raise NotImplementedError."""
    series = _make_series()
    diff = Diff(periods=1)
    with pytest.raises(NotImplementedError, match="weighted"):
        diff.transform(series, is_new_data=None, staleness_mode="weighted")


def test_weighted_staleness_mode_raises_on_chain():
    """staleness_mode='weighted' on ChainedTransform must also raise."""
    chain = ChainedTransform([Diff(periods=1), ZScore(window=10)])
    series = _make_series()
    with pytest.raises(NotImplementedError, match="weighted"):
        chain.transform(series, is_new_data=None, staleness_mode="weighted")
