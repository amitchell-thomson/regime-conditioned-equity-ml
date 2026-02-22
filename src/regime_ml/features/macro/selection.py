"""
Macro feature selection via within-group PCA.

Replaces the previous hardcoded ranked feature list with a data-driven approach:
PCA is applied independently within each macro group (rates, inflation, growth,
employment, liquidity, stress) using IS data only. The resulting PC columns are
the selected features for the HMM.

All selection parameters (PC counts per group, IS boundary) are configured in
configs/regimes/regime_config.yaml — no values are hardcoded here.
"""

from __future__ import annotations

import logging

import pandas as pd

from regime_ml.features.macro.group_pca import GroupPCATransformer

logger = logging.getLogger(__name__)


def select_features(
    features: pd.DataFrame,
    group_map: dict[str, str],
    cfg: dict,
) -> tuple[pd.DataFrame, GroupPCATransformer]:
    """
    Apply within-group PCA on IS data and return PC features for all dates.

    Reads configuration from cfg['regimes']:
        train_end_date                              — IS/OOS boundary (inclusive)
        feature_selection.group_pca.n_components    — {group: n_pcs} dict

    Args:
        features:  Full (IS + OOS) raw feature DataFrame with burn-in rows already
                   dropped (no NaNs). Produced by feature_data.dropna() in the pipeline.
        group_map: {feature_name: group} mapping, e.g. {'VIXCLS_level_zscore_63': 'stress'}.
                   Built by build_featuregroup_map().
        cfg:       Full config dict from load_configs().

    Returns:
        pc_features:  DataFrame indexed identically to `features`, columns like
                      rates_pc1, rates_pc2, stress_pc1, growth_pc1, ...
        transformer:  Fitted GroupPCATransformer. Use:
                          transformer.get_ordered_pc_columns()[:N]
                      to get the N most explanatory PCs for HMM candidate count N.
                          transformer.save_loadings(directory)
                      to persist loading matrices and explained variance.
    """
    regime_cfg = cfg.get("regimes", {})
    train_end_date: str = regime_cfg["train_end_date"]
    n_components: dict[str, int] = (
        regime_cfg["feature_selection"]["group_pca"]["n_components"]
    )

    logger.info(
        "Feature selection: within-group PCA | IS boundary: %s | groups: %s",
        train_end_date,
        {g: n for g, n in n_components.items()},
    )

    transformer = GroupPCATransformer(
        n_components_per_group=n_components,
        train_end_date=train_end_date,
    )
    pc_features = transformer.fit_transform(features, group_map)

    n_pcs = len(pc_features.columns)
    logger.info(
        "Selected %d PC features: %s",
        n_pcs,
        list(pc_features.columns),
    )

    return pc_features, transformer
