# 5 PCA Groups as the Feature Space

**Decision:** The HMM receives exactly 5 features — one PC1 per macro group: `rates_PC1`, `inflation_PC1`, `real_economy_PC1`, `credit_PC1`, `volatility_PC1`.

## Groups and series

| Group        | Series                              | PC1 sign anchor                                                          |
| ------------ | ----------------------------------- | ------------------------------------------------------------------------ |
| rates        | FEDFUNDS, DGS10, T10Y2Y, T10Y3M     | T10Y2Y (high=good: steep curve = accommodative)                          |
| inflation    | CPIAUCSL                            | none needed (positive YoY z-score = high inflation, correct orientation) |
| real_economy | CFNAI, INDPRO, ICSA, UNRATE, PAYEMS | CFNAI (high=good: strong activity)                                       |
| credit       | NFCI, BAA10Y                        | NFCI (low=good: easy conditions → flipped to positive)                   |
| volatility   | VIXCLS                              | VIXCLS (low=good: calm markets → flipped to positive)                    |

## Why sparse and grouped

**Interpretability:** Each PC1 maps to a single macro dimension (monetary stance, inflation, activity, credit stress, volatility). Regime transitions have a clear economic narrative.

**Redundancy elimination:** Within each group, series are correlated. PCA extracts the common factor and discards within-group noise. Example: DGS2 was dropped from rates because it is algebraically redundant given DGS10 and T10Y2Y.

**IS window preservation:** Series dropped from consideration due to short history (pre-2001 or pre-2004 start dates) — keeping only back-to-1985-capable series ensures the full IS window (1985→2019).

**Cross-group independence:** Post-PCA, cross-group PC pairs are monitored for correlation > 0.65. Separate groups for credit and volatility matters — VIX diverged from credit spreads in 2022 and 2015-16; merging them would lose those distinctions.

## Series explicitly excluded and why

| Series                           | Reason                                                                           |
| -------------------------------- | -------------------------------------------------------------------------------- |
| DGS2                             | Algebraically redundant (DGS10 - T10Y2Y = DGS2); no new information              |
| PCEPILFE, T10YIE, T5YIFR, AHETPI | Feature start 2004 — would truncate IS window                                    |
| JTSJOL                           | Feature start 2005 — was the binding IS-window constraint                        |
| UMCSENT                          | Feature start 2003; redundant with CFNAI composite                               |
| M2SL                             | r=-0.16 vs CPI in IS window; structurally different pre/post-QE (non-stationary) |
| BAMLH0A0HYM2                     | Feature start 2001; NFCI+BAA10Y sufficient for credit stress                     |
| WALCL                            | Feature start 2005; only meaningful post-QE era                                  |
