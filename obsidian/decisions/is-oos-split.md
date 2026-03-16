---
decision: IS/OOS split at 2019-01-01
date: pre-2026-03-16
status: settled — do not re-litigate
---

# Decision: IS/OOS Split at 2019-01-01

## Decision

`train_end_date: 2019-01-01` in `regime_config.yaml`. All data from 2019-01-01 onward is OOS.

## Rationale

The split is chosen so that all four major regime types appear out-of-sample:
- **COVID-19 crisis (2020)** — liquidity crisis / volatility shock
- **2022 inflation / Fed hiking cycle** — policy tightening regime
- **2023+ rapid disinflation** — late-cycle soft landing

An earlier split (e.g. 2015) would put these episodes in-sample, where the model can fit to them directly. The OOS period should contain diverse, challenging macro conditions that the model must generalise to.

**Why:** Without COVID and 2022 in OOS, selection and evaluation would only validate on relatively benign 2019 conditions. The OOS robustness score (weight=0.20) would be meaningless.

**How to apply:** IS = 2000-01-01 to 2018-12-31. OOS = 2019-01-01 onward. Never fit scalers, PCA, HMM, or thresholds on OOS data.
