# Evaluation Metrics

**Module:** `regimes/evaluation.py`

## Metrics used in model selection

| Metric | What it measures | Used as |
|---|---|---|
| Mahalanobis distance | Separation between regime centroids in feature space | Soft score (macro) |
| ANOVA R² | Fraction of feature variance explained by regime | Soft score (macro) |
| TV distance at n_mix | Total-variation distance of A^n_mix from stationary — how quickly regimes mix | Hard filter (`tv_distance_valid`) |
| Implied duration | Mean regime duration from diagonal of transition matrix | Soft score (transitions) |
| BIC | Bayesian Information Criterion — penalises complexity | Soft score (bic, weight=0.15) |
| OOS regime share | Fraction of each regime appearing in OOS period | Soft score (oos) |
| Label churn (CV) | Fraction of history relabeled as IS window expands | Hard filter (max_churn=0.65) |

## TV distance

Computed at `n_mix=20` trading days (~1 month). Catches absorbing states — a near-zero TV distance means the chain has mixed to its stationary distribution almost instantly, indicating an absorbing or near-absorbing state.

## Gaussian assumption validation

VIX and NFCI are winsorised at ±4σ before entering the HMM. This handles fat tails that would otherwise violate the Gaussian emission assumption and produce degenerate fits.

## Soft score formula

`_soft_score()` assigns 1.0 at `optimal`, 0.5 at `lo`/`hi` boundaries, decays to 0.0 at `lo - slack*(hi-lo)` and `hi + slack*(hi-lo)`. Thresholds in `regime_config.yaml` — never hardcoded.
