# Decisions

Architectural decisions and rationale. Captures the *why* behind design choices — things not obvious from reading the code.

## Index
- [[staleness-strict-default|Staleness strict mode as default]]
- [[fedfunds-monthly-frequency|FEDFUNDS corrected to monthly frequency]]
- [[five-indicator-set|5 PCA groups as the feature space]]
- [[kmeans-hmm-init|KMeans initialisation + multi-seed for HMM]]
- [[covariance-full|Full covariance only — diag excluded from HMM grid]]
- [[is-oos-split|IS/OOS split at 2019-01-01]]
- [[alfred-fred-routing|ALFRED vs FRED per-series routing]]
