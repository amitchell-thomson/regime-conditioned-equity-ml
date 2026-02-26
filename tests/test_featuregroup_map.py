"""Tests for build_featuregroup_map YAML-based implementation."""

from regime_ml.data.macro.build_featuregroup_map import build_featuregroup_map


def test_vixcls_maps_to_liquidity():
    """VIXCLS features must map to 'liquidity' as declared in regime_universe.yaml."""
    result = build_featuregroup_map(
        ["VIXCLS_level_zscore_63", "VIXCLS_diff_5_zscore_126"]
    )
    assert result["VIXCLS_level_zscore_63"] == "liquidity"
    assert result["VIXCLS_diff_5_zscore_126"] == "liquidity"


def test_known_series_map_correctly():
    """Spot-check that other series IDs resolve to their declared categories."""
    features = [
        "NFCI_level_zscore_50",  # liquidity
        "T10Y3M_level_zscore_252",  # rates
        "DGS2_level_zscore_252",  # rates
        "PCEPILFE_yoy_zscore_36",  # inflation
        "CFNAI_level_zscore_36",  # growth
        "INDPRO_yoy_zscore_36",  # growth
        "ICSA_ma_4_zscore_50",  # employment
    ]
    result = build_featuregroup_map(features)
    assert result["NFCI_level_zscore_50"] == "liquidity"
    assert result["T10Y3M_level_zscore_252"] == "rates"
    assert result["DGS2_level_zscore_252"] == "rates"
    assert result["PCEPILFE_yoy_zscore_36"] == "inflation"
    assert result["CFNAI_level_zscore_36"] == "growth"
    assert result["INDPRO_yoy_zscore_36"] == "growth"
    assert result["ICSA_ma_4_zscore_50"] == "employment"


def test_unknown_ticker_maps_to_unknown():
    """Feature names with unrecognised series IDs must map to 'unknown', not raise."""
    result = build_featuregroup_map(["NOTAREAL_zscore_63", "FAKE_diff_5"])
    assert result["NOTAREAL_zscore_63"] == "unknown"
    assert result["FAKE_diff_5"] == "unknown"


def test_empty_input_returns_empty_dict():
    """Empty input list must return empty dict."""
    assert build_featuregroup_map([]) == {}


def test_no_parquet_io(monkeypatch):
    """build_featuregroup_map must not call load_dataframe (no parquet I/O)."""
    import regime_ml.data.macro.build_featuregroup_map as mod

    # If load_dataframe is still imported and called, this test will catch it
    # by asserting the module no longer imports it
    assert not hasattr(
        mod, "load_dataframe"
    ), "build_featuregroup_map.py should not import load_dataframe after the refactor"
