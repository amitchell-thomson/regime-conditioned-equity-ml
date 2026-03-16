# Features

Transform chains applied to macro series, then reduced to PCA group factors for regime detection.

## Feature pipeline

Raw FRED series → transforms → 5 PCA group factors (one PC1 per group) → HMM input

## Key references

- [[macro-indicators]] — all 13 series: frequencies, ALFRED routing, staleness limits, sign anchoring
- [[pca-features]] — PCA construction, IS/OOS boundary, sign anchoring, cross-group correlation
- [[../decisions/five-indicator-set]] — why these 5 groups and which series were excluded

## Transform chain format (YAML)

```yaml
SERIESCODE:
  transforms:
    - [level, {z_score: {window: 126}}]
    - [{diff: {periods: 5}}, {z_score: {window: 126}}]
```

Windows are in **native frequency units** (trading days for daily, weeks for weekly, months for monthly) — not calendar days. This is handled automatically by `staleness_mode='strict'`.

## Feature naming convention

`{SERIES_ID}_{transform_chain}` e.g. `VIXCLS_level_zscore_126`, `T10Y2Y_diff_5_zscore_126`

## Key modules

- `features/common/transforms/` — BaseTransform, TransformRegistry, ChainedTransform
- `features/macro/group_pca.py` — GroupPCATransformer, sign anchoring, IS-only fitting
- `features/macro/pipeline.py` — apply transform chains, 11-point validation, PCA, save
