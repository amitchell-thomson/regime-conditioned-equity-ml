from regime_ml.utils.config import load_configs


def build_featuregroup_map(all_feature_names: list[str]) -> dict[str, str]:
    """Map feature names to macro group categories defined in regime_universe.yaml.

    Each series in configs/data/regime_universe.yaml declares a 'category' field
    (e.g. stress, liquidity, rates, inflation, growth, employment). This function
    reads that mapping and returns a dict from feature name to category.

    Feature names follow the convention '{SERIES_ID}_{transform_chain}', so the
    series ID is extracted as the prefix before the first underscore.

    Args:
        all_feature_names: List of engineered feature names, e.g.
            ["VIXCLS_level_zscore_63", "T10Y3M_diff_21_zscore_252"]

    Returns:
        Dict mapping each feature name to its category, e.g.
            {"VIXCLS_level_zscore_63": "stress", "T10Y3M_diff_21_zscore_252": "rates"}
        Features whose series ID is not found in config map to "unknown".
    """
    series_cfg = load_configs()["macro_data"]["regime_universe"]["series"]
    code_to_cat: dict[str, str] = {v["id"]: v["category"] for v in series_cfg.values()}
    return {
        feat: code_to_cat.get(feat.split("_")[0], "unknown")
        for feat in all_feature_names
    }
