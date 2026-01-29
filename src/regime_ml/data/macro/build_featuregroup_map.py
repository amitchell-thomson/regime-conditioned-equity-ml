from regime_ml.utils.config import load_configs
from regime_ml.data.common.loaders import load_dataframe

def build_featuregroup_map(all_feature_names: list[str]) -> dict[str, str]:
    macro_cfg = load_configs()["macro_data"]["regime_universe"]
    df_group = load_dataframe(macro_cfg["raw_path"])

    code_to_cat = dict(zip(df_group["series_code"], df_group["category"]))
    for code in code_to_cat.keys():
        if code == "VIXCLS":
            code_to_cat[code] = "stress"

    out: dict[str, str] = {}
    for feat in all_feature_names:
        ticker = feat.split("_")[0]  # e.g. "VIXCLS_level_zscore_63" -> "VIXCLS"
        out[feat] = code_to_cat.get(ticker, "unknown")
    return out