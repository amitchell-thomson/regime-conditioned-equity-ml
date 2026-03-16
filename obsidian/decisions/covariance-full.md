---
decision: covariance_type = 'full' only in HMM grid
date: pre-2026-03-16
status: settled — do not re-litigate
---

# Decision: Full Covariance Only

## Decision

Only `covariance_type='full'` is used in the HMM grid. `'diag'` is excluded from the search.

## Rationale

The 5 PCA input features are not orthogonal after PCA within groups. Cross-group correlations exist in the data — most importantly:
- `real_economy_pc1` vs `credit_pc1`: r ≈ -0.64 (employment and credit conditions move together but in opposing PC directions)

A diagonal covariance matrix assumes zero off-diagonal covariances — this misspecifies the joint distribution, producing poorly separated regimes. Full covariance correctly captures these correlations.

**Why:** Empirically tested — 'diag' models produce overlapping regime centroids and degenerate transition matrices. Full covariance is the correct model for this feature space.

**How to apply:** Only `covariance_types: ["full"]` in `regime_config.yaml` grid. Never add 'diag' to the grid.
