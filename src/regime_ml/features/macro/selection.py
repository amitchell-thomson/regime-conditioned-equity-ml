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

import numpy as np
import pandas as pd

from regime_ml.features.macro.group_pca import GroupPCATransformer

logger = logging.getLogger(__name__)


def _check_cross_group_correlation(
    pc_features: pd.DataFrame,
    train_end_date: str,
    warn_threshold: float,
) -> None:
    """Warn if any cross-group PC pair exceeds the correlation threshold on IS data.

    PCA makes within-group features orthogonal by construction, but two PCs from
    different macro groups can still be highly correlated. High cross-group correlation
    signals redundancy between groups that may reduce regime discriminability.

    This check is non-blocking (warning only) — correlated PCs may have distinct
    economic interpretations even when statistically correlated.

    Args:
        pc_features:    Full PC DataFrame (IS + OOS). Correlation is computed on IS only.
        train_end_date: IS/OOS boundary string. Rows with index <= this date are used.
        warn_threshold: Warn if any absolute Pearson correlation exceeds this value.
    """
    is_mask = pc_features.index <= pd.Timestamp(train_end_date)
    is_data = pc_features.loc[is_mask]

    if is_data.shape[0] < 2 or is_data.shape[1] < 2:
        return

    corr = is_data.corr().abs()
    np.fill_diagonal(corr.values, 0.0)

    cols = list(corr.columns)
    pairs = [
        (c1, c2, float(corr.loc[c1, c2]))
        for i, c1 in enumerate(cols)
        for c2 in cols[i + 1 :]
        if corr.loc[c1, c2] > warn_threshold
    ]
    if pairs:
        logger.warning(
            "Cross-group PC correlation exceeds %.2f for %d pair(s): %s. "
            "Consider reviewing group definitions or increasing n_components.",
            warn_threshold,
            len(pairs),
            [(c1, c2, f"{r:.3f}") for c1, c2, r in pairs],
        )


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
    n_components: dict[str, int] = regime_cfg["feature_selection"]["group_pca"]["n_components"]

    logger.info(
        "Feature selection: within-group PCA | IS boundary: %s | groups: %s",
        train_end_date,
        {g: n for g, n in n_components.items()},
    )

    sign_anchors: dict[str, dict] | None = (
        regime_cfg.get("feature_selection", {}).get("group_pca", {}).get("sign_anchors")
    )

    transformer = GroupPCATransformer(
        n_components_per_group=n_components,
        train_end_date=train_end_date,
        sign_anchors=sign_anchors,
    )
    pc_features = transformer.fit_transform(features, group_map)

    n_pcs = len(pc_features.columns)
    logger.info(
        "Selected %d PC features: %s",
        n_pcs,
        list(pc_features.columns),
    )

    # Check for high cross-group PC correlation (non-blocking warning).
    corr_cfg = regime_cfg.get("feature_selection", {}).get("cross_group_correlation", {})
    warn_threshold = float(corr_cfg.get("warn_threshold", 0.80))
    _check_cross_group_correlation(pc_features, train_end_date, warn_threshold)

    return pc_features, transformer
