# Macro Indicators Reference

All 13 series used in the pipeline. Defined in `configs/regime_universe.yaml`.

## Summary table

| Series | Category | Frequency | Source | ALFRED? | Role |
|---|---|---|---|---|---|
| VIXCLS | volatility | daily | FRED | No | CBOE VIX — market fear gauge |
| DGS10 | rates | daily | FRED | No | 10-year Treasury yield |
| T10Y2Y | rates | daily | FRED | No | 10Y minus 2Y spread (yield curve slope) |
| T10Y3M | rates | daily | FRED | No | 10Y minus 3-month spread |
| FEDFUNDS | rates | **monthly** | FRED | No | Effective federal funds rate |
| CPIAUCSL | inflation | monthly | FRED | Yes | Headline CPI (all urban consumers) |
| CFNAI | real_economy | monthly | FRED | Yes | Chicago Fed National Activity Index |
| INDPRO | real_economy | monthly | FRED | Yes | Industrial production index |
| ICSA | real_economy | **weekly** | FRED | Yes | Initial jobless claims |
| UNRATE | real_economy | monthly | FRED | Yes | Unemployment rate |
| PAYEMS | real_economy | monthly | FRED | Yes | Nonfarm payrolls |
| NFCI | credit | **weekly** | FRED | Yes | Chicago Fed National Financial Conditions Index |
| BAA10Y | credit | daily | FRED | No | BAA corporate - 10Y Treasury spread |

## Staleness limits (max before NaN-out)
- Daily: 5 business days
- Weekly: 21 business days (~1 month)
- Monthly: 65 business days (~3 months)

## PCA grouping and sign anchoring

Five groups are formed, one PC per group enters the HMM:

| Group | Series | PC1 positive = |
|---|---|---|
| rates | DGS10, T10Y2Y, T10Y3M, FEDFUNDS | Steep curve + low short rates (easing cycle) |
| inflation | CPIAUCSL | Above-target inflation |
| real_economy | CFNAI, INDPRO, ICSA, UNRATE, PAYEMS | Strong growth + tight labour |
| credit | NFCI, BAA10Y | Easy financial conditions (low NFCI, tight spreads) |
| volatility | VIXCLS | Calm markets (low VIX) |

Sign anchoring is applied in `group_pca.py` at fit time so positive always means "good conditions". Configuration in `regime_config.yaml` → `feature_selection.group_pca.sign_anchors`.

## Transform chains

All transforms are defined in `regime_universe.yaml`. The pattern is:
```yaml
SERIESCODE:
  transforms:
    - [level, {z_score: {window: 126}}]         # raw level z-scored
    - [{diff: {periods: 5}}, {z_score: {window: 126}}]  # 5-day diff then z-score
```

Transforms are staleness-aware by default (`staleness_mode='strict'`): rolling windows count actual observations, then results are forward-filled. Monthly FEDFUNDS z-scored with `window: 12` uses 12 actual monthly observations, not 12 calendar days.

## Special handling

- **VIXCLS, NFCI**: Winsorised at ±4σ before transforms (`winsorize_sigma: 4.0` in global config). These series have documented fat tails.
- **FEDFUNDS**: Monthly. Was incorrectly treated as daily in early versions — fixed in commit c726f35.
- **CFNAI**: Uses `use_alfred: false` despite being heavily revised in first 1-2 years after release. This is a documented voluntary concession — ALFRED vintage data for CFNAI would be more correct but acceptable approximation given revision magnitudes.

## Recommended additions

### PPIACO — Producer Price Index, All Commodities

**Status:** Proposed — not yet added

**Group:** inflation (add alongside CPIAUCSL)

**Why:**
- With only CPIAUCSL, the inflation PCA group is a trivially scaled scalar — no real PCA, no redundancy
- PPI provides supply-side inflation signal (cost-push) vs CPI's demand-pull blend
- PPI diverges from CPI in key regimes: 2015-16 (oil collapse → PPI fell, CPI stable), 2022 (PPI spiked first, then CPI), 2017 (both benign)
- Making inflation PC1 a genuine composite directly improves the 2022 vs 2017 vs 2019 discrimination that causes CV fold collapses

**Source:** FRED, monthly, starts 1913. No ALFRED needed (commodity price indices not heavily revised).

**Proposed transform:**
```yaml
ppi_all_commodities:
  id: PPIACO
  use_alfred: false
  category: inflation
  frequency: monthly
  transforms:
    - [{yoy: {periods: 12, method: pct_change}}, {z_score: {window: 36}}]
    # 36 actual monthly publications = 3 years of YoY context
```

**Sign anchoring:** inflation group sign anchor stays on CPIAUCSL. Positive inflation_pc1 = elevated CPI + PPI (inflationary). PPIACO negative loading on PC1 would flip the group sign — the sign anchor corrects this automatically.

**Config change:** add to `regime_universe.yaml` under `series:`; update `regime_config.yaml` `feature_selection.group_pca` if needed (still `inflation: 1` PC). No IS-window impact (PPIACO starts 1913).
