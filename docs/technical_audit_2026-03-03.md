# Regime Detection System — Improvement Plan & Technical Audit

**Date:** 2026-03-03
**Scope:** Full improvement roadmap to achieve a stable 5-state HMM regime predictor suitable for
Phase 3 conditioning. Covers CV stability, feature enrichment, initialisation strategy, archetype
refinement, and output infrastructure.
**Current winner:** model_2 (4-state, full-covariance, p_stay=0.99, BIC=73,681, score=0.869)
**Previous audit:** 2026-03-02 (open items from that audit drive this plan)
**Auditor:** Claude Sonnet 4.6

---

## Executive Summary

The regime detection infrastructure is sound. The remaining problems are entirely in the model
quality layer and trace to a single structural root cause: **the 4-state model cannot discriminate
the full range of macro environments seen in the IS window.** Mid-cycle expansion (2005-07) and
late-cycle slowdown (2019) are the two environments where the 4-state model completely loses
discriminative signal — these map directly to the four near-zero CV fold scores that inflate churn
to 0.643 and keep the system close to the hard-reject threshold.

A stable 5-state solution that separates `expansion` from `policy_constrained` resolves the
structural ambiguity. Getting there requires three parallel workstreams:

1. **Feature enrichment**: Make inflation and policy features more discriminating, particularly
   for mid-cycle environments where the current 4-feature inflation/rates representation is
   insufficiently differentiated.
2. **Surgical 5-state initialisation**: The existing KMeans-seeded EM cannot reliably find
   a stable 5-state solution from random initialisation. Initialising from the production 4-state
   model's means — splitting the most ambiguous state — dramatically narrows the search space.
3. **CV configuration tightening**: Once feature enrichment is in place, tighten the churn
   hard-filter and add a fold-collapse filter so only models with both low mean churn AND no
   near-zero folds can pass.

In parallel, two infrastructure improvements are needed regardless of model changes: fold-date
logging in CV output, and a detailed episode results file written to `data/regimes/`.

---

## 1. Root Cause Analysis: Near-Zero CV Fold Scores

### 1.1 Fold Date Reconstruction (New Finding)

The current CV runs 21 folds with `min_train_years=8`, `fold_step_months=12`, data starting ~1991.
Folds go **forward** (fold 0 = smallest IS, fold 20 = largest IS). Reconstructed fold dates:

| Fold | IS-end | OOS window | ANOVA R² | Economic context |
|------|--------|-----------|---------|-----------------|
| 0 | ~1999 | 1999→2000 | 0.071 | Late-90s Goldilocks expansion |
| **1** | **~2000** | **2000→2001** | **0.008** | **Dot-com bust onset** |
| 2 | ~2001 | 2001→2002 | 0.483 | Dot-com recession (discriminated well) |
| 4 | ~2003 | 2003→2004 | 0.484 | Mid-cycle expansion (discriminated well) |
| **6** | **~2005** | **2005→2006** | **0.008** | **Mid-cycle undifferentiated expansion** |
| **7** | **~2006** | **2006→2007** | **0.006** | **Pre-GFC expansion, credit bubble building** |
| 8 | ~2007 | 2007→2008 | 0.398 | GFC onset (discriminated well) |
| 11 | ~2010 | 2010→2011 | 0.456 | QE-era recovery (discriminated well) |
| 13 | ~2012 | 2012→2013 | 0.122 | Taper tantrum (low but not collapsed) |
| **19** | **~2018** | **2018→2019** | **0.008** | **2019 late-cycle slowdown** |

### 1.2 Diagnosis by Fold

**Fold 1 (2000-01 OOS — dot-com bust onset):** The model trained through 1999 has never observed
a technology crash. The dot-com bust is a genuinely novel OOS environment for this fold. This is
a structural "first-time" problem that no feature enrichment fully solves — the model cannot
learn what it has never seen. However, in production the model will always have the full IS
window (1991-2019) so fold 1 represents a pathological early-fold scenario unlikely to recur.
This fold contributes to churn but is the least actionable.

**Folds 6-7 (2005-07 OOS — mid-cycle expansion, boring tightening):** This is the most
diagnostically important finding. 2005-07 was a period of strong economic growth, gradually
tightening Fed policy (funds rate 2.5%→5.25%), low VIX, and tight IG spreads. With a 4-state
model that has only Risk-On, Contraction, Inflation/Policy, and Crisis, this period is
ambiguous: it is growth-positive (Risk-On direction) but also has a tightening policy stance
(Inflation/Policy direction). The model trained on IS data up to 2005-06 — which has not yet
seen the 2022 extreme policy tightening — finds no clean discriminating signal. The HMM
effectively flips between assigning 2005-07 to Risk-On or Inflation/Policy depending on the
random seed of the particular fold.

**The 5-state fix:** In a 5-state model with a dedicated `expansion` state (strong growth,
easy credit, calm VIX, normal-to-steepening curve) and a separate `policy_constrained` state
(tightening curve, above-target inflation, intact growth), 2005-07 maps cleanly to expansion.
The fold collapse disappears because the discrimination is no longer ambiguous.

**Fold 19 (2018-19 OOS — late-cycle slowdown):** The model trained through 2018 must classify
2018-19 OOS. The 2019 late-cycle slowdown had: yield curve inversion (T10Y3M went negative),
trade-war growth fears, tame inflation, intact labour market. In the 4-state model, this is
genuinely ambiguous: it is not Risk-On (inverted curve, elevated caution), not Inflation/Policy
(inflation is tame), not Crisis (no credit stress), and only borderline Contraction (growth still
positive). The HMM cannot assign it confidently, producing near-zero ANOVA R².

**The 5-state fix:** A `slowdown` / `late-cycle disinflation` state (inverted residual curve,
disinflation, near-zero growth, mild spread widening) perfectly matches 2019. This is the
canonical `slowdown` archetype already defined in the canonical pool — it just cannot manifest
in a 4-state model.

### 1.3 Implication for Churn

The four near-zero folds are the primary driver of `label_churn=0.643`. When consecutive folds
straddle a zero-R² fold, churn is nearly 1.0 (the coherent fold and the incoherent fold assign
completely different labels to the overlapping OOS window). If the four collapsed folds were
removed, the effective mean churn across the remaining 16-17 folds would fall substantially
— estimated below the 0.30 warning threshold.

**The path to low churn is not to tune the filter — it is to eliminate the folds where the
model has no discriminative signal.** That requires the 5-state model.

---

## 2. Prioritised Action Plan

### Priority 1 — Infrastructure: Fold-Date Logging and Episode Results File

These are pure additions with no risk of breaking anything. They should be done first so that
all subsequent runs produce richer diagnostics.

#### 1A. CV fold-date metadata

**File:** `src/regime_ml/regimes/evaluation.py`, `expanding_window_cv()`

The CV currently returns per-fold ANOVA R² scores as a flat list with no date annotation.
Inspecting near-zero scores requires off-line index reconstruction (as done above). Add a
`fold_metadata` list to the return dict:

```python
# Inside the fold loop, after computing fold_scores.append(...):
fold_metadata.append({
    "fold_index": len(fold_scores) - 1,
    "is_end": fold_end.date().isoformat(),
    "oos_start": fold_end.date().isoformat(),
    "oos_end": (fold_end + pd.DateOffset(months=oos_window_months)).date().isoformat(),
    "anova_r2": fold_scores[-1],
    "n_oos_obs": len(df_oos),
    "is_near_zero": fold_scores[-1] < 0.05 if np.isfinite(fold_scores[-1]) else None,
})
```

Add `fold_metadata: list[dict]` to the return dict and store in `run_metadata.json` under
`cv_diagnostics.fold_metadata`. The `oos_anova_r2_per_fold` list can be preserved for
backward compatibility but `fold_metadata` makes the list self-describing.

Config addition needed: none — all values are derived from existing config.

**Test:** Verify the fold metadata length matches `n_folds`. Verify `is_end` dates increase
monotonically. Verify `anova_r2` values match `fold_scores` list element-for-element.

#### 1B. Episode results output file

**File:** `src/regime_ml/regimes/pipeline.py` (or wherever `validate_against_episodes` is called)
**New output:** `data/regimes/episode_results.json`

Currently the episode validation result is truncated to `n_matched / n_episodes` in
`run_metadata.json`. The per-episode detail (match_status, expected_pct, archetype_match) is
not persisted to disk.

Write a dedicated `data/regimes/episode_results.json` containing the full per-episode breakdown:

```json
{
  "timestamp": "...",
  "best_model_id": "...",
  "pool": "canonical",
  "n_episodes": 16,
  "n_matched": 8,
  "n_not_in_model": 0,
  "n_applicable": 16,
  "episodes": [
    {
      "name": "2017 Expansion",
      "start": "2017-01-01",
      "end": "2017-12-31",
      "expected_archetype_canonical": "expansion",
      "expected_archetype_pool": "expansion",
      "match_status": "failed",
      "expected_pct": 0.12,
      "archetype_match": false,
      "dominant_state_label": "Policy Tightening",
      "dominant_state_pct": 0.73
    },
    ...
  ]
}
```

The `dominant_state_label` and `dominant_state_pct` fields show what the model actually
assigned during the episode — critical for diagnosing labeling failures.

With 5 canonical archetypes and 5 HMM states, `linear_sum_assignment` produces a square
assignment with no unmatched archetypes. All episodes have an applicable canonical archetype,
so `n_not_in_model` should be 0 for K≥5 runs using the canonical pool. For K=3 or K=4 runs,
`not_in_model` episodes may still occur where a coarsened pool key was not assigned to any
state (possible if the HMM fails to produce a state matching an archetype).

**Config addition:** Add `episode_results: "data/regimes/episode_results.json"` to the
`outputs:` block in `regime_config.yaml`. Never hardcode the path.

**Module ownership:** New output logic → `regimes/pipeline.py` (calls `validate_against_episodes`
and writes the JSON). The evaluation function itself should return the per-episode detail it
already computes internally.

---

### Priority 2 — Feature Enrichment: Inflation Group and Policy Rate

The current feature set has one inflation series (CPIAUCSL) and the FEDFUNDS window was already
fixed to 252 trading days. Two additions are recommended.

#### 2A. Add PPI to the inflation group

**File:** `configs/data/regime_universe.yaml`

With only CPIAUCSL, the inflation PCA group is trivially a scaled version of one series.
PPI (PPIACO — Producer Price Index, All Commodities) provides a supply-side inflation signal
that:
- Leads CPI by 1-3 months at turning points (useful for early detection)
- Diverges from CPI in 2015-16 (oil collapse → PPI fell sharply while CPI stayed stable)
- Diverges from CPI in 2022 (PPI spiked earlier and higher than CPI headline)
- Makes the inflation PC1 a genuine composite rather than a scaled scalar

**Series to add:**
```yaml
ppi_all_commodities:
  id: PPIACO
  use_alfred: false  # Producer price index; not heavily revised after initial release
  category: inflation
  frequency: monthly
  name: "Producer Price Index — All Commodities"
  description: "Supply-side inflation composite; leads CPI by 1-3 months at turning points.
                Diverges from CPI in commodity-driven regimes (2015-16 oil bust, 2022 energy)."
  transforms:
    - [{ yoy: { periods: 12, method: pct_change } }, { z_score: { window: 36 } }]
    # 36 calendar months (3 years) of actual publications
```

PPIACO is available from 1913; no IS-window constraint. With two inflation series, PCA will
produce a genuine PC1 (demand-pull vs cost-push blend) and the sign-anchor logic should
remain on CPIAUCSL (`good_direction: "low"` or add explicit no-anchor for inflation — since
elevated inflation is unambiguously negative for the economy, the current unsigned orientation
is actually correct and should not be flipped). Re-check sign anchoring logic when adding PPI.

**Expected improvement:** The inflation PC1 with PPI will discriminate 2015-16 (CPI stable,
PPI falling) from 2022 (CPI and PPI both surging) and from 2017 expansion (both mildly positive).
This directly improves the Inflation/Policy state's ability to repel expansion-era observations.

#### 2B. FEDFUNDS diff window semantics (already done — verify)

The diff window was changed from 6 days to 63 trading days (1 quarter). Verify the comment
now reads "Quarterly hiking/cutting cycle magnitude" and that the transform produces meaningful
values for:
- 2004-06 hiking cycle: should produce +z values as the Fed raised rates systematically
- 2007-08: near-zero diff (pause before cuts begin)
- 2018-19: positive diff, then abrupt reversal when the Fed pivoted

If the 252-day level z-score and 63-day diff together give a richer policy stance signal,
the rates PC1 should better discriminate 2005-07 (gradual tightening) from 2022 (shock
tightening) — helping the near-zero fold problem directly.

#### 2C. Consider adding T10YIE (5Y breakeven inflation)

**Conditional recommendation.** T10YIE starts 2004, which would compress the IS window to
2004+ for the inflation group's PCA fit. Given the data start is 1991 and the IS window
provides critical GFC/recovery data (2007-2011), this is a meaningful cost. **Do not add
T10YIE unless PPIACO proves insufficient.** If added, it would need to be in a separate group
or the IS-window impact must be explicitly accepted.

---

### Priority 3 — Surgical 5-State Initialisation

**This is the most impactful change for CV stability.** All five 5-state models currently fail
the CV churn hard filter (>0.65) because random KMeans seeds find different local optima for
5 states — the 5-state EM landscape is multimodal and the initialisation determines which
basin the algorithm converges to.

The surgical split approach bypasses this problem by starting from a known good 4-state
solution.

#### 3A. Implementation design

**New function in `src/regime_ml/regimes/hmm.py`:**

```python
def initialise_emissions_from_split(
    df_train: pd.DataFrame,
    base_detector: HMMRegimeDetector,
    base_scaler: StandardScaler,
    split_state: int,
    n_clusters_for_split: int = 2,
    random_state: int = 0,
    train_end_date: Optional[pd.Timestamp] = None,
    covariance_type: str = "full",
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Initialise a K+1 state HMM from a fitted K-state model by splitting one state.

    The K base states retain their means and covariances. The split_state is
    replaced by two sub-clusters found by K-Means on the observations assigned
    to that state. The resulting K+1 means and covariances are returned ready
    for HMMRegimeDetector initialisation.

    Args:
        df_train:         IS feature DataFrame.
        base_detector:    Fitted K-state HMMRegimeDetector.
        base_scaler:      StandardScaler used to transform df_train for base_detector.
        split_state:      Index of the state to split (0-indexed).
        n_clusters_for_split: Number of sub-clusters to create (default 2 → K+1 states).
        random_state:     KMeans seed for sub-cluster assignments.
        train_end_date:   IS boundary for validation.
        covariance_type:  Must match base_detector.covariance_type.

    Returns:
        (means, covs, scaler): Initialisation arrays for fit_best_of_n_seeds.
    """
```

**Algorithm:**
1. Predict Viterbi labels for all IS observations using the base model.
2. Extract all observations assigned to `split_state`.
3. Run KMeans(n_clusters=2) on those observations to produce two sub-clusters.
4. Compute Ledoit-Wolf covariance for each sub-cluster.
5. Assemble the K+1 means array: [base_means[:split_state], sub_mean_0, sub_mean_1, base_means[split_state+1:]]
6. Return with the same scaler (scaler does not change).

**Config addition:**
```yaml
# regime_config.yaml
hmm_grid:
  n_regimes: [4, 5]
  covariance_types: ["full"]
  p_stay: [0.95, 0.97, 0.99]

# New initialisation config
initialisation:
  # For n_regimes > 4, use surgical split from best n=K-1 model as a seeding strategy.
  # Reduces the search space for 5-state EM by anchoring 4 of the 5 states to a known
  # stable solution. The remaining seed budget (n_init - n_split_seeds) uses standard KMeans.
  use_surgical_split_for_n5: true
  n_surgical_seeds: 5         # How many split seeds to try (split each of the 4 states once)
  split_candidate_state: null # null = try all states; or specify one state index to split
```

**Pipeline integration in `pipeline.py`:**

The pipeline currently runs `compare_hmm_models()` which calls `fit_best_of_n_seeds()` for
each grid point independently. The surgical split requires a two-pass approach:

```
Pass 1: Fit all n=4 grid points (standard KMeans init, current code).
         → Select best n=4 model (becomes the "base" for pass 2).

Pass 2: Fit all n=5 grid points using:
         (a) n_surgical_seeds: initialise_emissions_from_split(base_4state, split_state)
             for each of the 4 possible states to split (gives 4 * p_stay_values seeds)
         (b) Remaining seeds: standard KMeans initialisation (as now)
         → Select best n=5 model with standard degeneracy + CV filters.

Final selection: Compare best n=4 and best n=5 models using select_best_hmm_model().
```

This requires modifying `compare_hmm_models()` or adding a new function
`fit_n5_with_surgical_split()` that accepts the best n=4 model as input.

**Which state to split?** The most ambiguous state in the current 4-state winner is state 2
(Inflation/Policy Stress) because it over-captures 2017 expansion and 2019 slowdown. This is
the natural split candidate: splitting it into `expansion` and `policy_constrained` (or
`slowdown`) creates exactly the 5-state structure needed.

However, trying all 4 states is better than hardcoding — the EM will converge to the right
solution even if initialised from a split of a state other than state 2, provided the
initialisation is close enough to the true 5-state solution.

#### 3B. Increase n_init for n=5 models

In addition to surgical split seeds, increase `n_init` from 10 to 20 for the 5-state fitting.
More standard KMeans seeds improve the chance of finding a stable EM basin independently.
The additional compute cost is one additional `fit_best_of_n_seeds()` call per grid point
(acceptable for an offline pipeline).

```yaml
hmm:
  n_init: 10          # Default for n=4 models
  n_init_n5: 20       # Extra seeds for n=5 models (more multimodal EM landscape)
```

---

### Priority 4 — CV Configuration: Fold-Collapse Filter and Threshold Tightening

#### 4A. Add fold-collapse hard filter

**File:** `configs/regimes/regime_config.yaml`

The current CV hard filter only checks mean churn (`max_churn: 0.65`). It does not penalise
models with multiple near-zero fold scores. A model can have mean churn=0.64 with 4 collapsed
folds and still pass.

Add a dedicated fold-collapse filter:

```yaml
cross_validation:
  hard_filter:
    max_churn: 0.50           # Tighten from 0.65 to 0.50 (see §4B below)
    min_cv_folds: 4
    max_zero_folds: 2         # NEW: reject if >2 folds have anova_r2 < zero_fold_threshold
    zero_fold_threshold: 0.05 # NEW: fold is "collapsed" if anova_r2 below this
```

**Implementation in `expanding_window_cv()`:**

```python
n_zero_folds = sum(1 for s in fold_scores if np.isfinite(s) and s < zero_fold_threshold)
fold_collapse_reject = bool(n_zero_folds > max_zero_folds)
churn_hard_reject = churn_hard_reject or fold_collapse_reject
```

Add `n_zero_folds` and `fold_collapse_reject` to the return dict for diagnostic visibility.

**Rationale:** A model with >2 zero-R² folds has structural discrimination failures in at
least 2-3 years of its IS history. Conditioning Phase 3 models on such a signal means
~10-15% of trading days will receive regime assignments from a state of near-random labeling.

#### 4B. Tighten mean churn threshold

After feature enrichment and surgical 5-state initialisation are in place, tighten the hard
churn threshold from 0.65 to 0.50. This is a separate config change that should only be made
after confirming the new 5-state model passes at the lower threshold.

**Do not tighten the threshold before the feature/initialisation work.** Tightening first
would cause the 3-state model (inherently lower churn from fewer states) to win by default,
which is the wrong outcome.

**Expected timeline:** Tighten to 0.50 after confirming a 5-state model passes 0.50.

#### 4C. Add churn soft-score penalty (medium priority)

The current design has a binary gap: `label_churn` is either below `max_churn_warning=0.20`
(no action) or above `max_churn: 0.65` (hard reject). Models in the 0.20-0.50 range get no
soft penalty.

```yaml
cross_validation:
  soft_score:
    enabled: true
    churn_soft_weight: 0.10   # Contribute 10% to final score
    churn_optimal: 0.10       # Score 1.0 at this churn level
    churn_lo: 0.05
    churn_hi: 0.25
    churn_slack: 1.0
```

This requires restoring churn as a soft-score component (it was removed when it was promoted
to a hard filter). The soft score only activates if `cross_validation.soft_score.enabled: true`.
This is optional and lower priority than the hard filter and threshold changes above.

---

### Priority 5 — Archetype Refinement for 5-State Model

When the 5-state model begins working, the canonical archetype pool (5 archetypes, used for
K≥5) will be in play. This is a square 5×5 linear assignment — every HMM state gets exactly
one archetype label with no unmatched archetypes. Two archetype signatures need adjustment
based on current labeling failures.

#### 5A. Tighten `slowdown` archetype to separate from `expansion`

The current `slowdown` (Late-Cycle Disinflation) archetype has:
```yaml
real_economy: 0.2    # near-zero growth
rates: -0.7          # mildly inverted
inflation: -1.0      # strong disinflation
```

The problem: in the 5-state model, both `expansion` (rates=+0.4) and `slowdown` (rates=-0.7)
must compete cleanly for 2017 vs 2019. The current `rates` gap of 1.1 is reasonable, but the
`real_economy` gap (expansion=+1.2 vs slowdown=+0.2) must carry most of the discrimination.
If 2019's real economy PC1 is closer to +0.5 than +0.2 (labour market was still intact),
the cosine distance to expansion will be small.

**Recommendation:** Reduce `slowdown.real_economy` to 0.0 (genuinely flat, not slightly positive)
and add a `rates: -0.8` to strengthen the inverted-curve signal. These values better reflect
the canonical late-cycle period where growth has flatlined but not turned negative.

Also remove `credit: -0.3` (moderate spread widening) — 2019 did not have meaningful
spread widening. Setting `credit: 0.0` makes the archetype more precisely the "everything-flat-
except-rates-inverted" environment.

#### 5B. Re-evaluate `liquidity_crisis` archetype extremity

State 3's confidence is 0.356 because the archetype requires `credit=-2.5` and `volatility=-2.5`
— GFC/COVID-level extremes. Moderate stress periods (2024-25) cannot match this archetype
even if they are genuinely below average in credit/volatility conditions.

This is being treated as a "Liquidity Crisis" labeling problem but the underlying issue is
that the 4-state model has no intermediate state. In the 5-state model, a period like 2024-25
that is slightly risk-off but not crisis can be assigned to `contraction` or `slowdown` rather
than being forced into crisis. The crisis archetype extremity can remain as-is for the 5-state
model — it correctly identifies GFC and COVID and nothing else, which is the right behaviour.

**Do not soften the crisis archetype to accommodate 2024-25.** Instead, verify after the
5-state run that 2024-25 is no longer dominated by the crisis state.

#### 5C. Canonical pool reduced to 5 archetypes (design decision)

**Decision:** The canonical archetype pool has been reduced from 7 to exactly 5 archetypes,
eliminating `stagflation` and `recession`. This makes the canonical pool square with the
5-state target model.

**Rationale:**
- `stagflation` had zero episodes in `economic_episodes.yaml` and no clear IS analog in the
  1991-2019 window. The 1970s oil shocks that define canonical stagflation are outside the
  data range. Including it created a poorly-anchored archetype that the labeling algorithm
  could accidentally assign to ambiguous states.
- `recession` had one episode (Dot-com Bust) and was empirically indistinguishable from
  `slowdown` in the macro PC space: both feature weak or flat real economy, disinflation,
  and moderate spread widening. The Dot-com Bust was a mild growth slowdown driven by tech
  sector correction — not a systemic recession with banking stress. Reclassified to `slowdown`.

**Remaining 5 canonical archetypes and their episode counts:**

| Archetype | Episodes | Description |
|-----------|---------|-------------|
| `expansion` | 4 | Strong growth, calm markets, normal curve |
| `recovery` | 3 | Post-crisis healing, steep curve, subdued inflation |
| `slowdown` | 5 | Late-cycle disinflation, flat growth, mild inversion |
| `policy_constrained` | 3 | Aggressive tightening, above-target inflation, intact growth |
| `liquidity_crisis` | 2 | Systemic credit stress, extreme volatility (GFC, COVID) |

With 5 archetypes and 5 HMM states, the linear assignment is square: every state maps to
exactly one archetype, every archetype maps to exactly one state, and `not_in_model` events
should not occur for K≥5 runs. The `episode_results.json` schema no longer needs an
`unmatched_archetypes` field.

The n3 and n4 pools update accordingly:
- n3 `contraction` blends `slowdown` + `policy_constrained` (inflation signals cancel)
- n4 `contraction` is pure `slowdown`; n4 `inflation_policy` is pure `policy_constrained`

#### 5D. Update n4 crisis archetype margin_warning logic

The current `Liquidity Crisis` margin_warning=False despite confidence=0.356. Looking at the
label config, `margin_warning_threshold: 0.08` checks the margin (0.122 > 0.08, so no warning).
But confidence=0.356 is a signal of labeling uncertainty that should also be surfaced.

Add a `min_confidence_for_warning` threshold to `label_config`:
```yaml
label_config:
  min_confidence: 0.18           # Hard floor for assigning a label
  margin_warning_threshold: 0.08 # Margin below which margin_warning=True
  confidence_warning_threshold: 0.50  # NEW: confidence below which confidence_warning=True
```

This allows downstream users to see which states are ambiguously labeled even when the margin
is technically adequate.

---

### Priority 6 — n4 Archetype Adjustments (for Current 4-State Model)

While pursuing the 5-state model, the following n4 pool adjustments improve the current
winner's episode validation score. These can be applied immediately.

#### 6A. Evaluate current archetype improvements

The 2026-03-02 audit prescribed two changes that were applied:
- `contraction.rates`: -0.2 → +0.2 (recession-weighted blend)
- `inflation_policy`: inflation 1.2 → 1.5, rates -1.0 → -1.3

These changes are already in `regime_archetypes.yaml`. **Re-run the full pipeline and verify
that episode validation improves from 8/16 before evaluating further n4 archetype changes.**
The current 8/16 score may already reflect these improvements; we need a fresh run to confirm.

#### 6B. Specific expected episode failures (for diagnosis)

Based on the label results and the fold-date analysis, the 8 likely failures are:

| Episode | Expected | Likely Assigned | Root Cause |
|---------|---------|----------------|-----------|
| 2017 Expansion | risk_on | inflation_policy | rates PC1 ≈ +1.5 in 2017 conflicts with inflation_policy.rates=-1.3; but if inflation signal is strong enough, still maps to inflation_policy |
| 2019 Late-Cycle Slowdown | contraction | risk_on or inflation_policy | No slowdown state in n4 pool |
| Dot-com Bust | contraction | unclear | rates PC1 ambiguous (early Fed cuts) |
| 2018 Late-Cycle Tightening | inflation_policy | risk_on | Pre-2022 tightening is mild compared to archetype |
| 2013 Taper Tantrum | risk_on | contraction | VIX spike + spread widening drags credit PC1 negative |
| 2024 Soft Landing | risk_on | crisis | OOS 2024-25 over-captures crisis state |
| 2025 Policy Uncertainty | contraction | crisis | Same as above |
| 2023 Disinflation Slowdown | contraction | inflation_policy | Disinflation with intact labour = ambiguous in n4 |

Seven of these eight failures can be attributed to the structural limitation of the 4-state
model. Only one (2018 Late-Cycle Tightening) might be fixable through archetype tuning alone.
This further confirms the 5-state model is the right path.

---

### Priority 7 — Outputs Configuration

Add the new output paths to `regime_config.yaml`:

```yaml
outputs:
  features_path: "data/features/macro_features_ready.parquet"
  regime_assignments: "data/regimes/regime_assignments.parquet"
  model_leaderboard: "data/regimes/model_leaderboard.csv"
  label_results: "data/regimes/label_results.json"
  run_metadata: "data/regimes/run_metadata.json"
  best_model: "data/regimes/best_model"
  episode_results: "data/regimes/episode_results.json"  # NEW — detailed per-episode validation
```

The `episode_results.json` path must be read from config in the pipeline — never hardcoded.

---

## 3. Implementation Sequence

The changes above should be applied in this order to enable clean validation of each step:

```
Step 1 (immediate, no pipeline re-run needed):
  - 1A: Add fold_metadata to expanding_window_cv() return dict
  - 1B: Add episode_results.json output
  - Priority 7: Add episode_results path to regime_config.yaml outputs
  → Validate: next pipeline run produces fold_metadata and episode_results.json

Step 2 (feature enrichment, requires pipeline re-run):
  - 2A: Add PPIACO to inflation group in regime_universe.yaml
  - Verify FEDFUNDS diff window is 63 trading days and comment is correct
  → Re-run full pipeline from feature construction
  → Validate: inflation PC1 loadings now show CPI and PPI contributions
              Episode validation score should improve (2017/2019 boundary)
              Fold scores for 2005-07 and 2019 windows should improve
              CV churn should fall from 0.643

Step 3 (surgical initialisation, requires hmm.py change):
  - 3A: Implement initialise_emissions_from_split() in hmm.py
  - 3B: Increase n_init_n5 to 20
  - Modify pipeline for two-pass 4-state → 5-state fitting
  → Re-run full pipeline
  → Validate: at least one 5-state model passes CV churn hard filter
              fold_score_slope should become positive (more IS data → better coherence)
              near-zero folds should disappear or reduce to ≤1

Step 4 (CV filter tightening, after confirming step 3 works):
  - 4A: Add fold-collapse filter (max_zero_folds: 2, zero_fold_threshold: 0.05)
  - 4B: Tighten max_churn to 0.50 (only after step 3 confirms a 5-state model passes)
  → Re-run pipeline
  → Validate: winning model passes both the new fold-collapse and tightened churn filters

Step 5 (archetype refinement, after 5-state model is stable):
  - 5A: Adjust slowdown archetype (already done: real_economy: 0.0, rates: -0.8, credit: 0.0)
  - 5D: Add confidence_warning_threshold to label_config
  → Re-run pipeline
  → Validate: episode validation ≥ 12/17 (now 17 applicable episodes with 5-archetype pool)
              All state confidences > 0.50
              No margin_warning or confidence_warning states (or justified exceptions)
```

---

## 4. Target State: Phase 3-Ready 5-State Model

The target regime signal for Phase 3 conditioning should have:

| Criterion | Current (4-state) | Target (5-state) |
|-----------|------------------|-----------------|
| CV churn | 0.643 (warning) | < 0.30 |
| Near-zero folds | 4 of 21 (19%) | 0-1 of 21 (< 5%) |
| Episode validation | 8/16 (50%) | ≥ 12/16 (75%) |
| Min state confidence | 0.356 (Crisis) | > 0.55 all states |
| Margin warnings | 1 (Contraction) | 0 |
| Max pairwise churn | 0.9482 | < 0.60 |

The 5-state model provides the state granularity Phase 3 models need:

| State | Phase 3 use |
|-------|------------|
| Expansion | Baseline bull regime — momentum, growth factors work |
| Recovery | Post-crisis / reflation — cyclical, value factors |
| Slowdown | Late-cycle / disinflation — defensive positioning |
| Policy-Constrained | Tightening regime — duration risk, quality factors |
| Liquidity Crisis | Crisis — tail risk, volatility premium |

Use `filter_proba_k` (not hard labels) for all Phase 3 conditioning. Soft probability vectors
allow the conditioning to gracefully interpolate across regime boundaries rather than creating
cliff effects at state transitions.

---

## 5. What Should Not Change

The following aspects of the system are correctly implemented and should not be touched:

1. **Staleness-strict transforms** — computing rolling statistics on real observations only is
   the single most important correctness property. This is not up for modification.
2. **IS-only scaler and PCA fitting** — correctly implemented with multiple layers of enforcement.
3. **`filter_proba` / `smooth_proba` separation** — the discipline at every boundary must be maintained.
4. **Ledoit-Wolf covariance in KMeans initialisation** — correct and well-motivated.
5. **Absolute soft-score thresholds for macro coherence** — solving the cross-n_regime percentile
   bias correctly. Do not reintroduce `rrank()` for macro or OOS scores.
6. **TV-20 as a hard filter only** — the tv_distance_valid check is correctly positioned.
   The tv_score soft component was correctly removed.
7. **Pool-routing labeling** (n3/n4/canonical) — elegant and maintainable. Only extends to
   accommodate 5-state usage of the canonical pool.

---

## 6. Open Items from Previous Audit

| Item | Status |
|------|--------|
| Near-zero CV fold scores: fold-date diagnosis | ✅ **Diagnosed** — folds 1, 6, 7, 19 identified as structural 4-state discrimination failures |
| CV reference fold → full IS model | ✅ **Done** (2026-03-02) |
| n4 contraction rates: -0.2 → +0.2 | ✅ **Done** (2026-03-02) |
| n4 inflation_policy: inflation 1.2→1.5, rates -1.0→-1.3 | ✅ **Done** (2026-03-02) |
| FEDFUNDS window comment mismatch | ✅ **Done** — window now 252 trading days |
| FEDFUNDS diff: 6-day window changed to 63 days | ✅ **Done** |
| `economic_episodes.yaml` stale 5-state comment | ✅ **Done** (2026-03-02) |
| `selection.py` default max_implied_duration mismatch | ✅ **Done** (2026-03-02) |
| Transform window unit ambiguity in YAML | ✅ **Done** (2026-03-02) |
| Inflation group: single series, trivial PCA | **→ Priority 2A** (add PPIACO) |
| CV churn: no intermediate action between warning (0.20) and hard-reject (0.65) | **→ Priority 4B/4C** |
| Near-zero fold scores: structural fix | **→ Priority 3 + 4** |
| episode_results.json output file | **→ Priority 1B** |
| CV fold-date metadata in run_metadata | **→ Priority 1A** |
| 5-state model CV instability | **→ Priority 3** (surgical split) |
| Canonical archetype pool: reduced to 5 archetypes, square 5×5 assignment | ✅ **Done** (2026-03-03) — `stagflation` and `recession` removed |
| Liquidity Crisis 2024-25 OOS dominance | **→ Revisit after step 3** |
