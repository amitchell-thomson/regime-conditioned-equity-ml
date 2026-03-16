# Phase 2 — Regime Detection

**Status:** 🔄 Nearly done — 2 structural blockers before Phase 3

## What it does
- HMM-based regime detection with KMeans initialisation
- Two-stage model selection: hard filters → soft weighted ranking
- Interpretable regime labeling matched to economic archetypes
- Causal (`filter_proba`) and non-causal (`smooth_proba`) probability outputs

## Key modules
- `regimes/hmm.py` — HMM detector (`filter_proba`, `smooth_proba`, `fit_best_of_n_seeds`)
- `regimes/evaluation.py` — metrics + `expanding_window_cv()`
- `regimes/selection.py` — two-stage selection (`select_best_hmm_model`)
- `regimes/labeling.py` — archetype matching (`label_regimes`)
- `regimes/pipeline.py` — 5-stage pipeline orchestrator

## Causal boundary
- `filter_proba()` — causal (online forward pass) — all trading signals
- `smooth_proba()` — non-causal (full history) — analysis/notebooks only

---

## Open blockers

### Blocker 1: CV churn — structural 4-state limitation

Current winner is 4-state. CV churn = 0.643 (just under the 0.65 hard filter).

**Root cause:** The 4-state model has no discriminative signal in:
- 2005-07 mid-cycle expansion (maps ambiguously to Risk-On or Inflation/Policy)
- 2019 late-cycle slowdown (matches none of the 4 states cleanly)

These produce 4 collapsed CV folds. The fix is a **stable 5-state model**, not threshold tuning.

**Fix path:** Surgical 5-state split initialisation — see [[../designs/five-state-surgical-split]]

See also: [[regimes/cross-validation]], [[regimes/known-issues#Issue 1]]

### Blocker 2: Archetype confidence scores

State 3 (Liquidity Crisis): confidence=0.356, barely above min_confidence=0.18.
State 0 (Contraction): margin_warning=True, runner-up is Liquidity Crisis.

**Root cause:** These are real diagnostic signals about the 4-state model's limitations, not labeling bugs. State 3 also over-dominates 2024-25 OOS — genuine mislabeling.

The 5-state model resolves both (cleaner state separation → higher confidence per state).

See also: [[regimes/known-issues#Issue 2]]

### Blocker 3 (secondary): Inflation group too thin

Only CPIAUCSL in inflation group — PCA is trivially a scaled scalar. No demand/supply distinction.

**Fix:** Add PPIACO to inflation group — see [[../features/macro-indicators#Recommended additions]]

---

## Implementation sequence

Order matters — later steps depend on earlier ones.

### Step 1 — Infrastructure (no pipeline re-run needed)
- Add `fold_metadata` (fold dates, anova_r2 per fold) to `expanding_window_cv()` return dict
- Add `episode_results.json` output (per-episode detail with dominant state labels)
- Add `episode_results` path to `regime_config.yaml` outputs block
→ Validate: next pipeline run produces fold_metadata and episode_results.json

### Step 2 — Feature enrichment (requires pipeline re-run from features)
- Add PPIACO to inflation group in `regime_universe.yaml`
- Verify FEDFUNDS diff window is 63 trading days with correct comment
→ Re-run from `regime-ml features`
→ Validate: inflation PC1 loadings show CPI+PPI contributions
→ Expected: CV churn falls from 0.643; fold scores for 2019 window improve

### Step 3 — Surgical 5-state init (requires hmm.py change)
- Implement `initialise_emissions_from_split()` in `hmm.py`
- Add `n_init_n5=20` to config
- Modify pipeline for two-pass 4-state → 5-state fitting
→ Re-run full pipeline
→ Validate: at least one 5-state model passes CV churn < 0.65
→ Target: folds 6, 7, 19 no longer near-zero; churn < 0.50

### Step 4 — CV filter tightening (after Step 3 confirmed)
- Add `max_zero_folds: 2` / `zero_fold_threshold: 0.05` to CV config
- Tighten `max_churn` from 0.65 to 0.50
→ Only after Step 3 confirms a 5-state model passes at 0.50

### Step 5 — Archetype refinement (after stable 5-state model)
- Verify `slowdown` archetype: `real_economy: 0.0`, `rates: -0.8`, `credit: 0.0` (already done)
- Add `confidence_warning_threshold: 0.50` to `label_config` in archetypes YAML
→ Validate: episode validation ≥ 12/17; all state confidences > 0.50

---

## Phase 3 readiness targets

| Metric | Current (4-state) | Target (5-state) |
|---|---|---|
| CV churn | 0.643 (warning) | < 0.30 |
| Near-zero folds | 4 of 21 (19%) | ≤ 1 of 21 |
| Episode validation | 8/16 (50%) | ≥ 12/17 (71%) |
| Min state confidence | 0.356 (Crisis) | > 0.55 all states |
| Margin warnings | 1 (Contraction) | 0 |
| Max pairwise churn | 0.9482 | < 0.60 |

## Phase 3 conditioning approach (pending)

Use `filter_proba_k` (not hard labels) for all conditioning. Soft probability vectors avoid cliff effects at regime boundaries. States with thin margins or low confidence (`filter_proba_k < 0.65`) should be treated as "mixed" rather than hard-assigned.

---

## Resolved

- ✅ CV reference model changed from first fold to full-IS model (more stable alignment anchor)
- ✅ n4 contraction archetype `rates`: −0.2 → +0.2 (avoids confusion with crisis archetype)
- ✅ n4 inflation_policy: inflation 1.2→1.5, rates −1.0→−1.3 (stronger tightening signal)
- ✅ FEDFUNDS diff window: 6 trading days → 63 days (actually captures cycle magnitude)
- ✅ tv_score removed from transition soft score (was 0 for all models — dead weight)
- ✅ CV selection gap fixed: CV now runs for all hard-filter-surviving models, not just top-6
- ✅ Canonical archetype pool reduced from 7 to exactly 5 (removed `stagflation`, `recession`) → square 5×5 linear assignment
- ✅ Labeling scheme generalises to fewer states — coarser labels for n3/n4 pools
- ✅ IS/OOS split at 2019-01-01 — see [[../decisions/is-oos-split]]
- ✅ covariance_type='full' only — see [[../decisions/covariance-full]]
- ✅ KMeans init + multi-seed — see [[../decisions/kmeans-hmm-init]]
- ✅ FEDFUNDS corrected to monthly frequency — see [[../decisions/fedfunds-monthly-frequency]]
