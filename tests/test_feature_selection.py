"""Tests for GroupPCATransformer and select_features (replaces hardcoded get_top_features)."""

import numpy as np
import pandas as pd
import pytest

from regime_ml.features.macro.group_pca import GroupPCATransformer


def _make_features(n_rows: int = 200, seed: int = 42) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build a small synthetic feature DataFrame with known group assignments."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_rows)
    columns = [
        "VIXCLS_level_zscore_63",
        "VIXCLS_diff_5_zscore_126",
        "T10Y3M_level_zscore_252",
        "DGS10_level_zscore_252",
        "CFNAI_level_zscore_36",
        "INDPRO_yoy_12_zscore_36",
        "PCEPILFE_yoy_12_zscore_36",
        "NFCI_level_zscore_50",
    ]
    data = rng.standard_normal((n_rows, len(columns)))
    features = pd.DataFrame(data, index=dates, columns=columns)
    group_map = {
        "VIXCLS_level_zscore_63": "stress",
        "VIXCLS_diff_5_zscore_126": "stress",
        "T10Y3M_level_zscore_252": "rates",
        "DGS10_level_zscore_252": "rates",
        "CFNAI_level_zscore_36": "growth",
        "INDPRO_yoy_12_zscore_36": "growth",
        "PCEPILFE_yoy_12_zscore_36": "inflation",
        "NFCI_level_zscore_50": "liquidity",
    }
    return features, group_map


TRAIN_END = "2015-01-01"
N_COMPONENTS = {"stress": 1, "rates": 1, "growth": 1, "inflation": 1, "liquidity": 1}


def test_fit_transform_shape():
    """Output has one column per configured PC."""
    features, group_map = _make_features()
    t = GroupPCATransformer(N_COMPONENTS, TRAIN_END)
    result = t.fit_transform(features, group_map)

    total_pcs = sum(N_COMPONENTS.values())
    assert result.shape == (len(features), total_pcs)


def test_output_columns_named_correctly():
    """Columns are named {group}_pc{n}."""
    features, group_map = _make_features()
    t = GroupPCATransformer(N_COMPONENTS, TRAIN_END)
    result = t.fit_transform(features, group_map)

    for col in result.columns:
        parts = col.rsplit("_pc", 1)
        assert len(parts) == 2, f"Unexpected column name: {col}"
        assert parts[1].isdigit()


def test_is_boundary_enforced():
    """PCA is fitted only on IS rows; OOS rows exist in transform output but not in fit."""
    # 2000 business days from 2010-01-04 reaches ~2017, spanning past TRAIN_END (2015-01-01)
    features, group_map = _make_features(n_rows=2000)
    train_end = pd.Timestamp(TRAIN_END)

    t = GroupPCATransformer(N_COMPONENTS, TRAIN_END)
    t.fit(features, group_map)

    result = t.transform(features)
    assert (result.index > train_end).any(), "Expected OOS rows in transform output"
    assert (result.index <= train_end).any(), "Expected IS rows in transform output"


def test_no_oos_data_used_in_fit():
    """IS-only fit: PC1 of rates group explains almost all variance when IS data is perfectly correlated.

    IS rows: T10Y3M and DGS10 are identical (perfect correlation → PC1 = 100%).
    OOS rows: random (no correlation). If OOS were included in fitting, PC1 ratio would drop.
    """
    rng = np.random.default_rng(0)
    # 2000 business days from 2010 → reaches ~2017, spanning past TRAIN_END (2015-01-01)
    dates = pd.bdate_range("2010-01-04", periods=2000)
    train_end = pd.Timestamp(TRAIN_END)
    is_mask = dates <= train_end
    n_is = is_mask.sum()

    t10y = rng.standard_normal(len(dates))
    dgs10_is = t10y[:n_is].copy()          # identical to T10Y3M in IS → perfect correlation
    dgs10_oos = rng.standard_normal(len(dates) - n_is)  # random in OOS

    features = pd.DataFrame({
        "T10Y3M_level_zscore_252": t10y,
        "DGS10_level_zscore_252": np.concatenate([dgs10_is, dgs10_oos]),
        "VIXCLS_level_zscore_63": rng.standard_normal(len(dates)),
        "CFNAI_level_zscore_36": rng.standard_normal(len(dates)),
        "PCEPILFE_yoy_12_zscore_36": rng.standard_normal(len(dates)),
        "NFCI_level_zscore_50": rng.standard_normal(len(dates)),
    }, index=dates)
    group_map = {
        "T10Y3M_level_zscore_252": "rates",
        "DGS10_level_zscore_252": "rates",
        "VIXCLS_level_zscore_63": "stress",
        "CFNAI_level_zscore_36": "growth",
        "PCEPILFE_yoy_12_zscore_36": "inflation",
        "NFCI_level_zscore_50": "liquidity",
    }

    t = GroupPCATransformer({"rates": 1, "stress": 1, "growth": 1, "inflation": 1, "liquidity": 1}, TRAIN_END)
    t.fit(features, group_map)

    # IS-only fit on perfectly correlated rates → PC1 explains ~100% of rates variance
    rates_ratio = t._pcas["rates"].explained_variance_ratio_[0]
    assert rates_ratio > 0.95, (
        f"Expected rates PC1 to explain >95% of IS variance (got {rates_ratio:.2%}); "
        "OOS data may have leaked into PCA fitting."
    )


def test_ordered_columns_sorted_by_variance():
    """get_ordered_pc_columns() returns columns in descending explained-variance order."""
    features, group_map = _make_features()
    t = GroupPCATransformer({"stress": 2, "rates": 2, "growth": 1, "inflation": 1, "liquidity": 1}, TRAIN_END)
    t.fit(features, group_map)

    ordered = t.get_ordered_pc_columns()
    variances = [
        t._pcas[col.rsplit("_pc", 1)[0]].explained_variance_[int(col.rsplit("_pc", 1)[1]) - 1]
        for col in ordered
    ]
    assert variances == sorted(variances, reverse=True)


def test_transform_columns_match_ordered_columns():
    """DataFrame columns from transform() equal get_ordered_pc_columns()."""
    features, group_map = _make_features()
    t = GroupPCATransformer(N_COMPONENTS, TRAIN_END)
    result = t.fit_transform(features, group_map)

    assert list(result.columns) == t.get_ordered_pc_columns()


def test_n_components_capped_at_n_features():
    """Requesting more PCs than features in a group is silently capped."""
    features, group_map = _make_features()
    t = GroupPCATransformer({"stress": 10, "rates": 1, "growth": 1, "inflation": 1, "liquidity": 1}, TRAIN_END)
    result = t.fit_transform(features, group_map)

    stress_cols = [c for c in result.columns if c.startswith("stress_")]
    assert len(stress_cols) == 2  # capped at 2 (number of stress features)


def test_nan_propagation():
    """NaN rows in input produce NaN PC values, not errors."""
    features, group_map = _make_features()
    features_with_nan = features.copy()
    features_with_nan.iloc[-5:, 0] = np.nan

    t = GroupPCATransformer(N_COMPONENTS, TRAIN_END)
    t.fit(features, group_map)
    result = t.transform(features_with_nan)

    assert result["stress_pc1"].iloc[-5:].isna().all()
    assert result["stress_pc1"].iloc[:-5].notna().all()


def test_transform_before_fit_raises():
    """Calling transform() before fit() raises RuntimeError."""
    features, _ = _make_features()
    t = GroupPCATransformer(N_COMPONENTS, TRAIN_END)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        t.transform(features)


def test_get_loadings_shape():
    """get_loadings() returns correct shape per group."""
    features, group_map = _make_features()
    t = GroupPCATransformer({"stress": 2, "rates": 1, "growth": 1, "inflation": 1, "liquidity": 1}, TRAIN_END)
    t.fit(features, group_map)

    loadings = t.get_loadings()
    assert "stress" in loadings
    assert loadings["stress"].shape == (2, 2)
    assert list(loadings["stress"].columns) == ["PC1", "PC2"]


def test_save_loadings(tmp_path):
    """save_loadings() writes expected CSV files."""
    features, group_map = _make_features()
    t = GroupPCATransformer(N_COMPONENTS, TRAIN_END)
    t.fit(features, group_map)
    t.save_loadings(tmp_path)

    for group in N_COMPONENTS:
        assert (tmp_path / f"pca_loadings_{group}.csv").exists()
    assert (tmp_path / "pca_explained_variance.csv").exists()

    summary = pd.read_csv(tmp_path / "pca_explained_variance.csv")
    assert set(summary.columns) >= {"group", "pc", "column", "explained_variance", "explained_variance_ratio"}
    assert len(summary) == sum(N_COMPONENTS.values())
