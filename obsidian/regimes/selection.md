# Model Selection

**Module:** `regimes/selection.py`
**Config:** `configs/regimes/regime_config.yaml` (selection section)

## Two-stage process

### Stage 1 — Hard filters (disqualifying)

Models failing any hard filter are rejected regardless of soft score:

| Filter | Threshold | Rationale |
|---|---|---|
| `max_implied_duration` | 1500 days | Catches near-absorbing states; soft score penalises >150d |
| `min_exit_paths` | 1 | Rejects truly absorbing states (0 exits) |
| `tv_distance_valid` | — | Rejects degenerate transition matrices |
| `min_regime_share` | 0.03 | Rejects models where any regime < 3% of history |
| `max_churn` (CV) | 0.65 | Rejects unstable models — relabeling >65% of history is disqualifying |
| `min_cv_folds` | 4 | Requires sufficient CV folds for a stable churn estimate |

### Stage 2 — Soft weighted ranking

Models passing hard filters are scored and ranked:

| Component | Weight | Metric |
|---|---|---|
| macro | 0.25 | Mahalanobis + ANOVA R² — IS feature coherence |
| transitions | 0.20 | Implied duration score + off-diagonal penalty |
| stability | 0.20 | Regime persistence |
| oos | 0.20 | OOS regime share balance |
| bic | 0.15 | BIC — primary discriminator when n_regimes differs |

**Churn is not a soft score** — it was moved to hard filter (edc95cc). The 0.15 weight previously on churn was redistributed: macro +0.05, oos +0.05, bic +0.05.

## Key design decisions

- **Absolute soft scores, not percentile ranks** for macro and oos: avoids bias toward fewer-regime models. A 3-regime and 4-regime model with similar absolute coherence receive similar scores; BIC then decides between them.
- **Turnover (TV-20) absent from soft score:** TV-20 values at p_stay in [0.93, 0.99] are 0.47-0.65, incompatible with former threshold. TV-20 is already a hard filter via `tv_distance_valid`.
