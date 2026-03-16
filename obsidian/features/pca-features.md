# PCA Feature Construction

**Module:** `features/macro/group_pca.py`, `features/macro/pipeline.py`
**Config:** `configs/regime_config.yaml` → `feature_selection.group_pca`

## Purpose

Reduce 13 raw indicators to 5 PCA features — one per macro group — before feeding the HMM. This balances:
- **Interpretability**: each PC has a clear economic meaning (growth, inflation, volatility, etc.)
- **Dimensionality**: 5 features avoids the curse of dimensionality in HMM fitting
- **Decorrelation**: within-group PCA removes redundant signal (e.g. T10Y2Y and T10Y3M are correlated)

## Fit boundary (IS/OOS)

PCA is fit **on IS data only** (`train_end_date=2019-01-01`). OOS data is transformed using IS-fit parameters. This is a hard constraint — fitting on OOS would introduce look-ahead bias.

## Sign anchoring

After fitting, PC1 directions are flipped if necessary so that **positive = economically good**. Sign anchors defined in config:

```yaml
feature_selection:
  group_pca:
    sign_anchors:
      rates: T10Y2Y        # positive loading → steep curve (good)
      inflation: CPIAUCSL  # positive loading → above target inflation (inflationary)
      real_economy: CFNAI  # positive loading → strong activity (good)
      credit: NFCI         # positive loading → NFCI flipped (low NFCI = good)
      volatility: VIXCLS   # positive loading → VIXCLS flipped (low VIX = good)
```

If the anchor series has a negative loading on PC1, the entire PC1 vector (loadings + scores) is negated.

## Cross-group correlation check

After fitting, pairwise correlations between PC groups are checked on IS data. If any pair > `warn_threshold=0.65`, a warning is logged. This does NOT fail the pipeline but signals that the PCA groups may not be as independent as assumed.

## Output columns

```
rates_pc1, inflation_pc1, real_economy_pc1, credit_pc1, volatility_pc1
```

Saved to `macro_features_ready.parquet`. Loadings saved to `pca_loadings.parquet`.

## Explained variance

PC1 typically explains 60-85% of within-group variance. If PC1 explains <50%, the group signal may be poorly represented by a single component — check loadings.

## Relationship to labeling

The archetype matching in `label_regimes()` uses these PC column names directly. The `featuregroup_map` fallback checks `feat.startswith(f"{group}_")` to map `rates_pc1` → `rates`. If this mapping fails, state vectors will have zeros for that group and confidence scores will be low.
