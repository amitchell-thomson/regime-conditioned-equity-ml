"""Tests for _check_cross_group_correlation() in features/macro/selection.py."""

import logging

import numpy as np
import pandas as pd
import pytest

from regime_ml.features.macro.selection import _check_cross_group_correlation


def _make_pc_features(
    n: int = 200,
    seed: int = 0,
    train_end_date: str = "2010-01-01",
) -> pd.DataFrame:
    """Create a synthetic PC feature DataFrame with IS + OOS rows."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    data = rng.standard_normal((n, 4))
    return pd.DataFrame(
        data,
        index=idx,
        columns=["rates_pc1", "inflation_pc1", "growth_pc1", "employment_pc1"],
    )


class TestCheckCrossGroupCorrelation:
    def test_no_warning_when_correlation_below_threshold(self, caplog):
        """No warning emitted when all cross-group PC pairs are below threshold."""
        rng = np.random.default_rng(99)
        idx = pd.bdate_range("2000-01-03", periods=300)
        # Independent columns → all pairwise correlations near 0
        df = pd.DataFrame(
            rng.standard_normal((300, 4)),
            index=idx,
            columns=["rates_pc1", "inflation_pc1", "growth_pc1", "employment_pc1"],
        )
        with caplog.at_level(logging.WARNING, logger="regime_ml"):
            _check_cross_group_correlation(df, "2010-01-01", warn_threshold=0.80)
        assert not any("Cross-group" in r.message for r in caplog.records)

    def test_warning_emitted_when_correlation_above_threshold(self, caplog):
        """Warning logged when two columns are perfectly correlated on IS data."""
        idx = pd.bdate_range("2000-01-03", periods=300)
        base = np.random.default_rng(7).standard_normal(300)
        df = pd.DataFrame(
            {
                "rates_pc1": base,
                "inflation_pc1": base,  # perfectly correlated with rates_pc1
                "growth_pc1": base * -1.0,  # perfectly anti-correlated
                "employment_pc1": np.random.default_rng(8).standard_normal(300),
            },
            index=idx,
        )
        with caplog.at_level(logging.WARNING, logger="regime_ml"):
            _check_cross_group_correlation(df, "2015-01-01", warn_threshold=0.80)
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any(
            "Cross-group" in m for m in warning_messages
        ), "Expected a cross-group correlation warning"

    def test_uses_is_data_only(self, caplog):
        """Correlation is computed on IS rows only; OOS high correlation must not trigger warning."""
        rng = np.random.default_rng(42)
        n = 400
        train_end_date = "2005-01-01"
        idx = pd.bdate_range("2000-01-03", periods=n)

        # IS portion: uncorrelated (all before train_end_date)
        data = rng.standard_normal((n, 4))
        df = pd.DataFrame(
            data,
            index=idx,
            columns=["rates_pc1", "inflation_pc1", "growth_pc1", "employment_pc1"],
        )

        # Artificially make OOS rows perfectly correlated in two columns
        oos_mask = df.index > pd.Timestamp(train_end_date)
        df.loc[oos_mask, "inflation_pc1"] = df.loc[oos_mask, "rates_pc1"]

        with caplog.at_level(logging.WARNING, logger="regime_ml"):
            _check_cross_group_correlation(df, train_end_date, warn_threshold=0.80)
        # OOS-only correlation must NOT trigger a warning
        assert not any("Cross-group" in r.message for r in caplog.records)

    def test_returns_none_when_insufficient_data(self):
        """Function returns None gracefully when IS data has fewer than 2 rows."""
        idx = pd.bdate_range("2000-01-03", periods=1)
        df = pd.DataFrame({"rates_pc1": [1.0], "inflation_pc1": [2.0]}, index=idx)
        result = _check_cross_group_correlation(df, "2010-01-01", warn_threshold=0.80)
        assert result is None

    def test_returns_none_for_single_column(self):
        """Function returns None gracefully when only one PC column is present."""
        idx = pd.bdate_range("2000-01-03", periods=100)
        df = pd.DataFrame(
            {"rates_pc1": np.random.default_rng(0).standard_normal(100)}, index=idx
        )
        result = _check_cross_group_correlation(df, "2005-01-01", warn_threshold=0.80)
        assert result is None
