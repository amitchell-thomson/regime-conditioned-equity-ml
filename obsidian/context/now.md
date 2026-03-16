# Current Context

**Last updated:** 2026-03-16

## Where we are

- ✅ Phase 1 — Data pipeline complete
- 🔄 Phase 2 — Regime detection **nearly done — 2 structural blockers**
- ⏳ Phase 3 — Regime-conditioned models: not started (blocked on Phase 2)
- ⏳ Phase 4 — Portfolio evaluation: not started

## Phase 2 open blockers

### Blocker 1: CV churn — 4-state structural limitation
`label_churn=0.643` for the current 4-state winner (threshold=0.65 — barely passing).

**Root cause (diagnosed):** Four CV folds collapse to near-zero ANOVA R² because the 4-state model cannot discriminate mid-cycle expansion (2005-07) or late-cycle slowdown (2019) — no clean 4-state archetype exists for either.

**Fix:** Stable 5-state model via surgical split initialisation → [[designs/five-state-surgical-split]]

**Next step:** Step 1 (add fold_metadata to CV output), then Step 2 (add PPIACO), then Step 3 (surgical split).

Full implementation sequence: [[phases/phase-2-regime-detection#Implementation sequence]]

### Blocker 2: Archetype confidence scores
State 3 (Liquidity Crisis): confidence=0.356. State 0 (Contraction): margin_warning, runner-up is Crisis.

**Root cause:** Real diagnostic signals about 4-state limitations. State 3 also over-dominates 2024-25 OOS (genuine mislabeling).

**Fix:** 5-state model resolves both (cleaner separation → higher confidence). Verify `featuregroup_map` fallback for PCA columns in the meantime.

Full details: [[regimes/known-issues]]

## Implementation sequence for Phase 2 completion

1. Add fold_metadata to CV output (no re-run)
2. Add PPIACO to inflation group, re-run features + regime
3. Implement surgical 5-state split → re-run pipeline
4. Tighten CV filters once 5-state model passes
5. Archetype refinement validation

## Phase 3 readiness targets

| Metric | Current | Target |
|---|---|---|
| CV churn | 0.643 | < 0.30 |
| Near-zero CV folds | 4 of 21 | ≤ 1 |
| Episode validation | 8/16 | ≥ 12/17 |
| Min state confidence | 0.356 | > 0.55 |
| Max pairwise churn | 0.9482 | < 0.60 |

## Recently settled decisions

- ✅ TV-score removed from transition soft score (was 0 for all models)
- ✅ CV selection gap fixed: CV runs for all hard-filter survivors, not just top-6
- ✅ Canonical archetype pool reduced to exactly 5 (removed stagflation, recession)
- ✅ CV reference changed from first fold to full-IS model
- ✅ n4 archetype signatures strengthened (contraction rates +0.2; inflation_policy inflation 1.5, rates -1.3)
- ✅ FEDFUNDS diff window: 6 → 63 trading days
- ✅ covariance_type='full' only → [[decisions/covariance-full]]
- ✅ IS/OOS split at 2019-01-01 → [[decisions/is-oos-split]]
- ✅ ALFRED routing per series → [[decisions/alfred-fred-routing]]
- ✅ FEDFUNDS corrected to monthly → [[decisions/fedfunds-monthly-frequency]]
- ✅ 5 PCA groups confirmed → [[decisions/five-indicator-set]]
- ✅ KMeans init + multi-seed → [[decisions/kmeans-hmm-init]]
- ✅ staleness_mode='strict' as default → [[decisions/staleness-strict-default]]

## Do not re-litigate

- PCA group structure (5 groups: rates, inflation, real_economy, credit, volatility) is fixed
- KMeans HMM initialisation is fixed
- IS/OOS split at 2019-01-01 is fixed
- covariance_type='full' only — 'diag' misspecifies cross-group PCA correlations
- Absolute soft-score thresholds for macro/OOS (not percentile rank) — solves cross-n_regime bias

## Pending design proposals

- [[designs/five-state-surgical-split]] — awaiting implementation approval

## Constraints reminder

- `src/regime_ml/models/` does not exist — do not import
- `src/regime_ml/portfolio/` does not exist — do not import
- Any change to regime inference or evaluation → write design proposal first, do not implement
- `smooth_proba()` is analysis-only — never in trading signals
