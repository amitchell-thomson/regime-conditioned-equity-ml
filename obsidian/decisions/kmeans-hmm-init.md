# KMeans Initialisation for HMM

**Decision:** HMM emission means are initialised using KMeans cluster centres rather than random initialisation. Multi-seed fitting (`n_init=10`) tries seeds `range(0, 10)` and selects the highest-log-likelihood model that passes the degeneracy filter.

## Why KMeans init

Random EM initialisation for HMMs is highly sensitive to starting conditions and frequently converges to degenerate solutions (absorbing states, near-empty regimes). KMeans provides a geometrically meaningful starting point: cluster centres roughly correspond to regime centroids in feature space, giving EM a better basin to converge from.

## Why multi-seed

Even with KMeans init, 5-regime models previously converged to the same degenerate EM solution across all seeds (all seeds → identical LL, producing an absorbing state). Multi-seed fitting with degeneracy checking (`tv_distance_valid` per seed) allows the selector to skip degenerate candidates and find a valid solution if one exists.

## Degeneracy filter

A model is rejected at the seed level if:
- Any regime has share < `min_regime_share` (0.03) — catches near-empty states
- `tv_distance_valid` fails — catches absorbing states (TV distance at mixing horizon ≈ 0)

The best non-degenerate seed by log-likelihood is returned.

## Config

```yaml
hmm:
  n_init: 10
  min_regime_share: 0.03
```
