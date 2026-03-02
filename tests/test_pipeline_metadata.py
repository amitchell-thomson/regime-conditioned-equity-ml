"""Tests for pipeline determinism — features_hash added to run_metadata."""

import hashlib

import numpy as np
import pandas as pd


def _make_features(n: int = 50, d: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(
        rng.standard_normal((n, d)),
        index=idx,
        columns=[f"pc{i}" for i in range(d)],
    )


def _compute_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


class TestFeaturesHash:
    def test_hash_is_16_char_hex(self):
        """features_hash must be a 16-character hexadecimal string."""
        df = _make_features()
        h = _compute_hash(df)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_stable_across_calls(self):
        """Same DataFrame produces the same hash on repeated calls."""
        df = _make_features(seed=42)
        assert _compute_hash(df) == _compute_hash(df)

    def test_hash_changes_when_data_changes(self):
        """Modifying a single value changes the hash."""
        df1 = _make_features(seed=7)
        df2 = df1.copy()
        df2.iloc[0, 0] += 1.0
        assert _compute_hash(df1) != _compute_hash(df2)

    def test_hash_changes_when_index_changes(self):
        """pd.util.hash_pandas_object includes the index; shifting it changes the hash."""
        df1 = _make_features(seed=3)
        df2 = df1.copy()
        df2.index = pd.bdate_range("2001-01-02", periods=len(df2))
        assert _compute_hash(df1) != _compute_hash(df2)

    def test_hash_changes_when_column_added(self):
        """Adding a column (i.e. upstream feature change) changes the hash."""
        df1 = _make_features(seed=5)
        df2 = df1.copy()
        df2["extra"] = 0.0
        assert _compute_hash(df1) != _compute_hash(df2)
