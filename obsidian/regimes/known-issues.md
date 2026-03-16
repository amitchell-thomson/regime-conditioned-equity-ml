# Known Issues — Regime Detection

**Status as of:** 2026-03-16
**Phase:** 2 (nearly complete — 2 blockers)

---

## Issue 1: CV scores are terrible

### Root cause (diagnosed)

The near-zero CV fold scores are a **structural 4-state problem**, not a bug.

The 4-state model cannot discriminate:
- **Mid-cycle expansion (2005-07)**: strong growth, gradual Fed tightening (2.5%→5.25%), low VIX, tight spreads. With only Risk-On / Contraction / Inflation-Policy / Crisis states, 2005-07 sits ambiguously between Risk-On (growth positive) and Inflation-Policy (tightening). Folds whose OOS window covers this period flip randomly between the two states.
- **2019 late-cycle slowdown**: yield curve inverted (T10Y3M went negative), tame inflation, intact labour market, no credit stress. This maps to none of the 4 states cleanly — not Risk-On, not Inflation-Policy (inflation tame), not Crisis (no credit stress), only borderline Contraction (growth still positive).

**In a 5-state model** these separate cleanly: `expansion` (rates +0.4) captures 2005-07; `slowdown` (inverted curve, disinflation, flat growth) captures 2019.

### Fold-date reconstruction (from 03-03 audit)

The 4-state model ran 21 folds with `min_train_years=8`, `fold_step_months=12`:

| Fold | IS-end | Near zero? | Economic context |
|---|---|---|---|
| 0 | ~1999 | Low R² | Late-90s expansion — limited macro dispersion |
| **1** | **~2000** | **~0.008** | **Dot-com bust onset — model never seen a tech crash** |
| 2 | ~2001 | OK | Dot-com recession — discriminated well |
| **6** | **~2005** | **~0.008** | **Mid-cycle expansion — no discriminative signal in 4-state** |
| **7** | **~2006** | **~0.006** | **Pre-GFC expansion with credit bubble building** |
| 8 | ~2007 | OK | GFC onset — discriminated well |
| 11 | ~2010 | OK | QE recovery — discriminated well |
| **19** | **~2018** | **~0.008** | **2019 late-cycle slowdown — no clean 4-state match** |

The 4 collapsed folds drive `label_churn=0.643`. Without them, effective mean churn is estimated well below 0.30. **The path to low churn is a stable 5-state model, not tuning the filter.**

### CV metrics from 03-02 audit (4-state winner)

- `label_churn`: 0.6147 (warning, threshold = 0.65)
- `max_pairwise_churn`: 0.9482 — worst consecutive fold pair churned 94.8% of assignments
- `n_nontrivial_perms`: 23/24 folds required state reordering
- `fold_score_slope`: -0.00067 (flat — coherence does not improve with more IS data)

### Fix: surgical 5-state split initialisation

See [[../designs/five-state-surgical-split]] for full algorithm design.

Summary: KMeans seeds find different local optima for 5-state EM. Initialise from the production 4-state model's means — splitting the most ambiguous state (state 2, Inflation/Policy) into two sub-clusters. This anchors 4 of 5 states to a known stable solution and dramatically narrows the EM search space.

### Secondary fix: fold-collapse hard filter

After surgical split is in place, add `max_zero_folds: 2` to CV config. A model with >2 collapsed folds should be hard-rejected regardless of mean churn.

---

## Issue 2: Regime archetype confidence scores — per-state assessment

### Current 4-state confidence (from 03-02 audit)

| State | Label | Confidence | Margin | Status |
|---|---|---|---|---|
| 0 | Contraction | 0.645 | 0.070 | ⚠️ margin_warning; runner-up is Liquidity Crisis |
| 1 | Risk-On | 0.953 | 1.600 | ✅ clean |
| 2 | Inflation / Policy Stress | 0.898 | 1.041 | ⚠️ over-captures 2017/2019 (see §structural) |
| 3 | Liquidity Crisis | 0.356 | 0.122 | ❌ barely above min_confidence |

### State 0 — Contraction / Liquidity Crisis confusion

Root cause: the n4 `contraction` archetype rates signature is near-neutral (−0.2), which blends slowdown (rates=−0.7) and recession (rates=+0.3 due to emergency Fed cuts). The cancellation makes contraction ambiguous against crisis (rates=+1.5) when the model's contraction state has even mild curve steepening. **Fixed in archetype YAML (rates changed to +0.2).**

### State 2 — Inflation/Policy over-reach

Root cause: n4 `inflation_policy` rates=−1.0 is not strong enough to exclusively attract 2022 (rates_pc1≈−2.8) while repelling 2017 expansion (rates_pc1≈+1.5). The blended archetype sits between the two, attracting both. **Fixed in archetype YAML (inflation 1.2→1.5, rates −1.0→−1.3). Root cause is 4-state compression — 5-state model fully resolves this.**

### State 3 — Liquidity Crisis confidence=0.356

Root cause: the crisis archetype requires extreme values (credit=−2.5, vol=−2.5 = GFC/COVID-level). State 3 may represent "moderate financial stress" with actual feature means around credit=−1.0, vol=−1.0. The cosine similarity to the extreme archetype is low by construction.

Additionally, state 3 over-dominates 2024-25 OOS (26.9% of OOS days). This is genuine mis-labeling — 2024-25 is not a liquidity crisis. The 5-state model should place 2024-25 in a different state.

**For Phase 3:** use `filter_proba_3` (soft probability) not hard label. Apply `filter_proba_3 > 0.70` gate — observations where the crisis state is ambiguous should not be force-assigned.

### State 1 — Risk-On

Clean. Confidence 0.953, margin 1.600. Safe for hard conditioning in Phase 3.

### Algorithmic fix for confidence scores

The core labeling algorithm is correct (cosine similarity + linear assignment). Low confidence scores for states 3 and 0 are **real diagnostic signals about the model**, not bugs in the labeling code. The fix is the 5-state model.

The one potential code-level issue: verify `featuregroup_map` correctly maps PCA columns (`rates_pc1` etc.) to groups. If the fallback fails, state_vecs will have zeros for missing groups and confidence scores will be artificially deflated. See [[labeling#featuregroup_map fallback]].

---

## Issue 3: Single-series inflation group

With only CPIAUCSL in the inflation group, PCA is trivially a scaled version of one series. No redundancy and no ability to distinguish demand-pull (CPI-driven) from cost-push (PPI-driven) inflation.

**Recommended fix:** Add `PPIACO` (Producer Price Index, All Commodities) to the inflation group. See [[../features/macro-indicators#Recommended additions]].

This directly improves the 2022 vs 2015-16 vs 2017 discrimination — PPI spiked earlier in 2022, fell sharply in 2015-16 while CPI stayed stable, and was benign in 2017. The inflation PC1 becomes a genuine composite.
