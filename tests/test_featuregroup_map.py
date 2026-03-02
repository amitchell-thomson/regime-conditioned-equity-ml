"""Tests for build_featuregroup_map YAML-based implementation."""

from regime_ml.data.macro.build_featuregroup_map import build_featuregroup_map


def test_vixcls_maps_to_volatility():
    """VIXCLS features must map to 'volatility' as declared in regime_universe.yaml."""
    result = build_featuregroup_map(["VIXCLS_level_zscore_126", "VIXCLS_level_zscore_63"])
    assert result["VIXCLS_level_zscore_126"] == "volatility"
    assert result["VIXCLS_level_zscore_63"] == "volatility"


def test_known_series_map_correctly():
    """Spot-check that series IDs resolve to their declared categories.

    Group structure (post-restructure):
      rates        — FEDFUNDS, DGS10, T10Y2Y, T10Y3M
      inflation    — CPIAUCSL
      real_economy — CFNAI, INDPRO, ICSA, UNRATE, PAYEMS
      credit       — NFCI, BAA10Y
      volatility   — VIXCLS
    Dropped series (DGS2, PCEPILFE, etc.) map to 'unknown'.
    """
    features = [
        "NFCI_level_zscore_50",  # credit
        "T10Y3M_level_zscore_252",  # rates
        "CPIAUCSL_yoy_zscore_36",  # inflation
        "CFNAI_level_zscore_36",  # real_economy
        "INDPRO_yoy_zscore_36",  # real_economy
        "ICSA_ma_4_zscore_50",  # real_economy
        "UNRATE_level_zscore_36",  # real_economy
        "PAYEMS_yoy_zscore_36",  # real_economy
        "BAA10Y_level_zscore_252",  # credit
        "VIXCLS_level_zscore_126",  # volatility
        "DGS2_level_zscore_252",  # dropped → unknown
    ]
    result = build_featuregroup_map(features)
    assert result["NFCI_level_zscore_50"] == "credit"
    assert result["T10Y3M_level_zscore_252"] == "rates"
    assert result["CPIAUCSL_yoy_zscore_36"] == "inflation"
    assert result["CFNAI_level_zscore_36"] == "real_economy"
    assert result["INDPRO_yoy_zscore_36"] == "real_economy"
    assert result["ICSA_ma_4_zscore_50"] == "real_economy"
    assert result["UNRATE_level_zscore_36"] == "real_economy"
    assert result["PAYEMS_yoy_zscore_36"] == "real_economy"
    assert result["BAA10Y_level_zscore_252"] == "credit"
    assert result["VIXCLS_level_zscore_126"] == "volatility"
    assert result["DGS2_level_zscore_252"] == "unknown"


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
