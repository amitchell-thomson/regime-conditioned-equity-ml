# Regimes

HMM-based market regime detection.

## Regime types (canonical 5-archetype pool)
- Expansion
- Recovery
- Late-Cycle Disinflation
- Policy Tightening
- Liquidity Crisis

(4-state: Risk-On / Contraction / Inflation Policy / Liquidity Crisis)
(3-state: Risk-On / Contraction / Liquidity Crisis)

## Key concepts
- [[hmm-model|HMM Model]] — architecture, initialisation, IS/OOS split
- [[labeling|Labeling Scheme]] — archetype matching, economic episode validation
- [[evaluation|Evaluation Metrics]] — TV distance, Mahalanobis, BIC, churn
- [[selection|Model Selection]] — two-stage hard filter + soft weighted ranking
- [[cross-validation|Cross-Validation]] — expanding-window CV, label churn hard filter
- [[known-issues|Known Issues]] — CV scores, confidence score bugs (Phase 2 blockers)

## Causal boundary (critical)
| Function | Causal? | Use in |
|---|---|---|
| `filter_proba()` | Yes | Trading signals, all production logic |
| `smooth_proba()` | No | Analysis, notebooks only |
