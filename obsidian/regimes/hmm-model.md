# HMM Model

**Module:** `regimes/hmm.py`

## Architecture

Gaussian HMM with full covariance. Receives 5 PCA features (one per macro group) — see [[../decisions/five-indicator-set]].

**Why full covariance only:** Cross-group PCA correlations exist (e.g. employment-growth r=-0.64). Diagonal covariance misspecifies the joint distribution and produces poorly separated regimes.

## Initialisation

KMeans initialisation with multi-seed fitting. See [[../decisions/kmeans-hmm-init]].

## Grid search parameters

```yaml
hmm_grid:
  n_regimes: [4, 5]
  covariance_types: ["full"]
  p_stay: [0.95, 0.97, 0.99]
```

## Probability outputs

| Method | Causal? | Use in |
|---|---|---|
| `filter_proba()` | Yes — online forward pass | All trading signals and production logic |
| `smooth_proba()` | No — uses full history | Analysis and notebooks only |

**Never use `smooth_proba()` in anything that feeds a trading decision.**

## IS/OOS split

`train_end_date: 2019-01-01` — keeps COVID (2020), 2022 inflation shock, and 2023+ slowdown in OOS, ensuring all four regime types appear out-of-sample.
