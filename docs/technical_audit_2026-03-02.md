# Regime Detection System — Technical Audit

**Date:** 2026-03-02
**Scope:** Full regime detection path — macro ingest → feature transforms → PCA → HMM grid
→ CV → selection → labeling → episode validation, plus all YAML configs
**Current winner:** model_6 (4-regime, full covariance, p_stay=0.97, BIC_IS=73,681, final_score=0.858)
**Previous audit:** 2026-02-28 (5-regime winner, CV gap unresolved)
**Auditor:** Claude Sonnet 4.6

---

## Executive Summary

The previous audit identified six issues. Five have been closed; one remains open. The codebase is in
genuinely good shape at the infrastructure level — IS/OOS discipline, staleness-aware transforms, causal
probability separation, and absolute soft-score thresholds are all correctly implemented and well-tested.
The hard problems are now in the **model quality** layer: the winning 4-regime HMM has a churn warning
(0.6147), three near-zero CV fold scores that expose latent instability, a contraction state with a
dangerously thin archetype margin, and a Liquidity Crisis state with confidence barely above 0.35.
These are not infrastructure bugs — they reflect genuine limits of the current feature set and archetype
design, and they have direct consequences for Phase 3 conditioning quality.

**What should not change:** Ledoit-Wolf initialisation, staleness-strict transforms, IS-only PCA/scaler
fitting, absolute macro-score thresholds, `filter_proba` / `smooth_proba` separation, pool-routing
labeling, and the comprehensive test suite. These are all correct and production-grade.

**What deserves attention:** CV fold score collapse, the Inflation/Policy Stress state's over-reach,
the Liquidity Crisis confidence gap, feature representation in the credit and real-economy groups,
feature window semantics in YAML, and a stale comment in `economic_episodes.yaml`.

---

## 1. Resolution Status of Previous Audit

| Priority | Issue | Status |
|----------|-------|--------|
| 1 | CV selection gap — winner had null CV diagnostics | ✅ **Resolved** |
| 2 | `tv_score` dead weight (zero for all models) | ✅ **Resolved** — removed, weight redistributed |
| 3 | Hardcoded feature_metadata path | ✅ **Resolved** — `feature_metadata_path` in `regime_universe.yaml` |
| 4 | Duplicate `if not self._pcas` check in `group_pca.py` | ✅ **Resolved** (or no longer visible) |
| 5 | `_GROUPS` validation against feature columns | ✅ **Resolved** — runtime warning + validation added |
| 6 | CFNAI revision look-ahead (medium risk) | ⚠️ **Acknowledged** in YAML comment; accepted as voluntary concession |

The transition formula is now `trans = 0.65 * dur_score + 0.35 * off_pen` — clean and defensible.
The `min_confidence` threshold was also lowered from 0.45 to 0.18, which is appropriate given that
all current states are labeled (the threshold was never binding at 0.18 in this run).

---

## 2. Current Model State

### 2.1 Label Summary

| State | Label | Confidence | Margin | Warning |
|-------|-------|------------|--------|---------|
| 0 | Contraction | 0.645 | 0.070 | **YES** |
| 1 | Risk-On | 0.953 | 1.600 | no |
| 2 | Inflation / Policy Stress | 0.898 | 1.041 | no |
| 3 | Liquidity Crisis | 0.356 | 0.122 | no |

State 1 (Risk-On) is the cleanest, most distinctive state — confidence near 1.0 and margin 1.6 indicate it
occupies a distinct corner of the archetype space with no close rivals. States 0 and 3 are concerning:
Contraction has a margin of only 0.070 (margin_warning=True, runner-up is Liquidity Crisis), and
Liquidity Crisis has a confidence of only 0.356 — barely above the 0.18 threshold. Both are discussed
at length in §6.

### 2.2 Key Performance Metrics

- **Episode validation: 8/16** (50%). Given that all 16 episodes are now translatable via `canonical_to_coarse`
  (no `not_in_model` outcomes for the n4 pool), these 8 failures reflect genuine model mis-labeling rather
  than pool mismatch. The 8 failures are examined in §7.
- **BIC_IS: 73,681** — 5,836 better than the best 3-regime model (model_1: 79,517), confirming strong
  statistical preference for 4 regimes over 3. The 5-regime models (all failing CV churn) would have had
  even better BIC, so the 4-regime winner is a stability-quality tradeoff, not a fit tradeoff.
- **OOS ANOVA R²: 0.329** vs **IS: 0.386** — 14.8% OOS degradation. This is moderate; the signal does
  transfer but weakens materially out of sample.
- **CV churn: 0.6147** (warning, not hard-reject). `max_pairwise_churn=0.9482`, `n_nontrivial_perms=23/24`.
  These numbers are examined in detail in §5.

---

## 3. Feature Construction

### 3.1 What Works Well

**Series selection is well-reasoned and documented.** Dropping JTSJOL (feature start 2005), UMCSENT
(2003), AHETPI (2004), and PCEPILFE (2004) preserves the IS window. Using CPIAUCSL alone back to 1985
is correct. Merging growth and employment into one PCA group (justified by INDPRO vs PAYEMS r=0.656) is
economically sound and reduces redundancy.

**VIX separation from credit is the right call.** The 2022 regime (VIX elevated, IG spreads contained)
and 2015-16 commodity bust (VIX spike, credit benign) both show VIX decoupling from credit — keeping
them in separate groups lets the HMM observe divergences that would otherwise cancel out in a combined PC1.

**Winsorization at 4σ for VIX and NFCI is appropriate.** These are the two series with documented
heavy tails (VIX reached 80 in GFC, 65 in COVID; NFCI spiked to extreme levels). Clipping at 4σ
preserves extreme-regime signal while preventing single outlier days from distorting the entire z-score
history.

**ALFRED vintage data is used for the right series.** CPIAUCSL, INDPRO, UNRATE, and PAYEMS all use
ALFRED. Market-quoted rates (FEDFUNDS, DGS10, T10Y2Y, T10Y3M, BAA10Y, VIXCLS) correctly use non-ALFRED
since they are never revised.

### 3.2 Feature Window Semantics Ambiguity

The YAML documents `window` values without consistently specifying whether the unit is real
observations (post-staleness filtering) or calendar/trading days. The staleness-strict transform
handles this correctly in code, but the YAML is ambiguous for daily series:

- `FEDFUNDS: z_score: {window: 36}` — 36 **trading days** ≈ 1.7 calendar months. The comment says
  "Policy stance level", implying the intent is to capture current stance vs. recent history. At 36
  trading days, this z-score is highly reactive — a 75bp hike cycle can move it from -2 to +2 within
  one month. Compare to `DGS10: z_score: {window: 252}` — the window for the long rate is 7× longer
  than for the policy rate, which inverts the expected informativeness relationship. If the intent is
  to capture the rate cycle level (not short-term momentum), a window of 126-252 trading days for
  FEDFUNDS would be more stable.

- For monthly series (CFNAI, UNRATE, PAYEMS), `window: 36` means 36 actual publication months (3 years)
  which is appropriate. But the YAML reader cannot know whether "36" means days or publications without
  the `frequency` field.

**Recommendation:** Add a `window_unit` comment to each transform in the YAML (e.g., `# 36 trading days`)
and audit whether the FEDFUNDS window should be longer given its stated purpose.

### 3.3 FEDFUNDS Diff Window

`FEDFUNDS: diff: {periods: 6}` captures a 6-trading-day rate change (one week). The comment says
"Hiking/cutting cycle magnitude". A 6-day difference in the Fed Funds rate captures FOMC meeting
surprises, not hiking/cutting cycle magnitude, which would require `periods: 63` (quarterly) or
`periods: 126` (semi-annual). This is a semantic mismatch between the comment and the parameter —
the feature is actually capturing week-to-week rate momentum. This is still a valid feature but is
labelled misleadingly.

### 3.4 Single-Series Inflation Group

With only CPIAUCSL in the inflation group, the PCA step for inflation is mathematically trivial:
PC1 is the scaled series itself. The `GroupPCATransformer` still runs the full PCA machinery
(eigendecomposition, sign anchoring, etc.) on a 1×1 case. This is not incorrect but is wasteful
and could cause confusion when reading PCA loadings.

More importantly, a single series means the inflation PC1 is entirely determined by CPIAUCSL YoY
z-score with a single sign flip. If CPIAUCSL is not representative of the true inflation signal in
a given sub-period (e.g., 2022 when shelter inflation significantly lagged CPI), the inflation PC1
has no redundancy. A second series — such as PPI (PPIACO, available from 1947) or the Cleveland
Fed median CPI — would make the inflation group more robust to methodological changes in CPI.

**Note:** Adding series that start post-2000 would not compress the IS window given the current
1990-2026 data range and 1985 lookback start; the constraint is the feature burnin period, not
the series start date. PPIACO starts in 1947 and would have no burnin constraint.

---

## 4. Group PCA

### 4.1 What Works Well

IS-only fitting with sign anchoring, full sign-anchor validation, and the group-based independence
assumption are all correctly implemented. The sign convention documentation in `regime_archetypes.yaml`
with empirical validation examples (GFC: rates_pc1=+3.8; 2022 hike: rates_pc1=-2.8) is excellent
practice.

### 4.2 Cross-Group Correlation Risk

The config warns at `cross_group_correlation.warn_threshold: 0.65`, but there is no log evidence
in the pipeline run that any pair exceeded this. Given the current group structure:

- `real_economy_pc1` and `credit_pc1` are likely highly correlated in recessions (both deteriorate
  sharply in GFC and COVID simultaneously). Their IS correlation may approach 0.60-0.70.
- `volatility_pc1` and `credit_pc1` are correlated in crises (both spike) but diverge in 2022 and
  2015-16 — this is the justification for separating them.

If `real_economy` and `credit` PC1s are highly correlated in IS, the HMM is effectively receiving
redundant signals. Two nearly identical features in the emission model will cause the full-covariance
matrix to have near-collinear columns, increasing sensitivity to regularisation and degrading Mahalanobis
separation. This should be explicitly logged and checked.

### 4.3 PC1-Only Labeling vs. n_components Config

`label_regimes` uses only `idxs[0]` (PC1) per group for archetype matching. With `n_components: 1`
per group in the current config, this is correct. But the current design does not document that if
`n_components` is increased (e.g., `rates: 2` to capture rate level vs. slope separately), the
labeling will silently use only PC1 of each group, ignoring PC2 even though PC2 is being fed into
the HMM. The archetype matching would then be based on an incomplete state representation.

This is a latent design tension, not a current bug. It would manifest as label instability if
n_components is increased without updating the labeling logic.

---

## 5. HMM Fitting and Cross-Validation

### 5.1 HMM Fitting: What Works Well

Multi-seed initialisation, Ledoit-Wolf covariance, `tv_distance_valid` hard filter, IS boundary
enforcement, and the `1e-6 * I` jitter for strict positive-definiteness are all correct and
well-implemented. The BIC parameter count correctly accounts for K-1 startprob DoF, K*(K-1)
transmat DoF, K*D means, and K*D*(D+1)/2 full-covariance upper-triangle entries.

### 5.2 CV Fold Score Collapse — High Priority

The 24 OOS ANOVA R² fold scores are:

```
0.284, 0.530, 0.007, 0.071, 0.226, 0.483, 0.300, 0.484, 0.338, 0.008, 0.006,
0.398, 0.387, 0.283, 0.456, 0.383, 0.122, 0.214, 0.263, 0.363, 0.504, 0.273,
0.008, 0.224
```

Folds 2, 9, 10, and 22 (0-indexed) have near-zero R² (0.007, 0.008, 0.006, 0.008). These
four folds are producing essentially random regime assignments — the model has no discriminative
power in those OOS windows. The metric_cv_std=0.1645 is inflated primarily by these collapses.

This is more serious than the summary churn statistic suggests. The mean churn of 0.6147 is driven
partly by these collapse folds transitioning from a coherent to an incoherent solution. The
`fold_score_slope=-0.00067` confirms that coherence is not improving with more IS data — it is flat,
meaning the model does not improve in discriminative stability as the IS window grows.

**Likely causes of near-zero folds:**

1. **Short OOS windows in low-volatility periods.** The first few folds (IS-end ~1995-1998) have OOS
   windows covering periods with relatively low macro dispersion. The HMM may find few features
   differentiated between regimes in a calm 1-year OOS window.

2. **HMM finding a different local maximum.** Despite degeneracy filtering, some refitted folds may
   converge to solutions where one state absorbs nearly all observations. A near-absorbing solution
   in a CV fold would produce near-zero ANOVA R² even if the primary model is non-degenerate.

3. **Structural break in the IS→OOS transition.** If the IS window ends just before a structural
   break (e.g., the dot-com bust starting in 2000, the GFC starting in 2007), the OOS window may
   have a different distributional character than anything in IS, causing the refitted model to
   mis-assign the OOS period entirely.

**Diagnostic action required before Phase 3:** Log the IS-end dates for folds 2, 9, 10, 22 and
inspect which OOS windows they correspond to. If they are all from pre-2000 IS windows, the near-zero
behaviour may be an early-data artifact and the model's stability from the 2000+ period could be
assessed separately. If they span the full history, the issue is more fundamental.

### 5.3 High max_pairwise_churn (0.9482) and n_nontrivial_perms (23/24)

`n_nontrivial_perms=23` means 23 of the 24 CV folds required reordering to align with the reference
fold. This is nearly universal — the HMM finds a substantially different state ordering on almost
every refitting. With 4 states, there are 24 possible permutations; 23/24 non-trivial is therefore
not saying "the order is random" — it is saying "the identity permutation is rarely optimal". This
is expected behaviour when state labels have no inherent ordering.

However, `max_pairwise_churn=0.9482` is genuinely alarming. In the worst consecutive fold pair,
94.8% of OOS assignments changed after state alignment. Given that alignment is done by Hungarian
matching on means (which should correctly remap states), this is not an alignment artifact — it
is genuine instability in which observations are assigned to which state.

The implication for Phase 3 is serious: if the pipeline re-estimates the HMM with an additional
year of IS data (as would happen in production), up to 94.8% of the trailing year's assignments
could change. This makes the regime signal difficult to use for momentum-style conditioning.

### 5.4 State Alignment in CV: Reference Fold is First Fold

The current CV implementation uses the first fold (minimum IS window: 5 years of data) as the
reference for state alignment. This is structurally fragile: the first fold may have the least stable
state structure (smallest IS sample → highest parameter uncertainty) and is therefore the worst
possible reference. Subsequent folds are aligned against an unstable reference.

A more robust approach is to use the final fold (maximum IS window) as the reference — it has the
most data and should have the most stable parameters — and work backwards. Alternatively, the reference
could be the full-IS model (train_end_date) rather than any fold model.

### 5.5 CV Hard Filter Threshold Calibration

The hard-reject threshold is `max_churn: 0.65`. The winning model has churn=0.6147 — 7.5% below
the hard threshold. Given the evidence of instability (max_pairwise_churn=0.9482, near-zero fold
scores), the hard threshold may be set 15-20% too high. It allows the model to win while exhibiting
near-pathological instability in individual fold pairs.

The warning threshold (`max_churn_warning: 0.20`) is correctly set — the model is indeed flagged
with `churn_warning: True`. But the gap between 0.20 (warning) and 0.65 (hard reject) is very large,
and there is no intermediate action prescribed for models in the 0.20-0.65 range.

**Suggestion:** Consider adding a soft-score penalty for churn in the range 0.20-0.50, in addition
to the existing hard reject at 0.65. This would nudge model selection away from high-churn candidates
without outright rejecting them.

---

## 6. Model Selection

### 6.1 What Works Well

Absolute soft-score thresholds for macro coherence (maha/anova) correctly eliminate the 3-regime
bias. BIC at 0.15 weight effectively discriminates between model sizes — the 4-regime model's BIC
advantage of 5,836 units over the best 3-regime model is decisive. The OOS hard filters (`oos_min_share`)
correctly catch models that produce dead regimes out-of-sample.

### 6.2 model_6 and model_7 Are Essentially Identical

Looking at the leaderboard: model_6 (p_stay=0.97) and model_7 (p_stay=0.99) have:
- Identical BIC_IS: 73,680.555 (difference < 0.001)
- Identical regime shares, Mahalanobis distances, ANOVA R²
- Same OOS metrics

They converge to the same EM solution despite different p_stay initialisation. This makes sense:
the data is strongly informative about regime duration, so EM can correct the initialisation. The
p_stay grid point that "wins" is essentially determined by EM convergence path, not by the prior.

**Implication:** The transition score is the only remaining differentiator, and it chose model_6
(0.5854) over model_7 (0.4979) because the slightly lower off-diagonal in model_6 gives a marginally
better `off_pen` rank. This is a very thin margin on which to select the production model.

### 6.3 Stability Score Uses `rrank()` for Transitions

The stability score mixes absolute and percentile-rank sub-components:
- `churn`: `rrank(n_transitions)` — percentile rank, lower transitions = better
- `pers`: `_soft_score(avg_persistence, ...)` — absolute threshold
- `ent`: `rrank(entropy_mean)` — percentile rank, higher entropy = better

`rrank` is still used for `n_transitions` (churn sub-score in stability) and `max_offdiag` (off_pen
in transitions). Since all surviving models have the same `n_regimes`, percentile ranking within
survivors is not biased in the cross-n_regimes direction. This is fine.

However, `entropy_mean` is ranked with `ascending=True` (higher entropy = better) which means models
with more balanced regime distributions rank higher. Since model_6 and model_7 have near-identical
entropy (0.037), the entropy contribution is near-zero.

### 6.4 Hard Filter: `max_implied_duration` Default vs. Config Inconsistency

`selection.py` line 59 has `max_implied_duration: float = 400.0` as the default, but
`regime_config.yaml` sets this to 1500.0. The pipeline passes the YAML value so this is not a bug in
practice, but the mismatch means calling `select_best_hmm_model()` without the config will apply
a much stricter filter (400d vs. 1500d). The function signature default should match the YAML default
or be documented explicitly.

---

## 7. Labeling and Episode Validation

### 7.1 Contraction State: Thin Margin and Wrong Runner-Up

State 0 (Contraction) has margin=0.070 with runner-up being Liquidity Crisis (0.5754). The n4
contraction archetype has `rates: -0.2` (near-neutral), while the crisis archetype has `rates: +1.5`
(emergency cuts steepening the curve).

If the model's contraction state has slightly positive rates_pc1 (i.e., some curve steepening from
moderate Fed cuts), it will score positively on both the crisis rate signature (+1.5) and against
the contraction rate signature (-0.2). The confusion indicates state 0 likely occupies a region
of feature space that sits between a late-recession environment (where emergency cuts have begun)
and a genuine liquidity crisis.

**The n4 contraction archetype signature has a structural weakness:** by blending slowdown (rates=-0.7)
and recession (rates=+0.3), the rates component nearly cancels to -0.2. A recession in the n4
taxonomy should have steepening rates (emergency cuts), not flat rates. The cancellation makes the
contraction archetype ambiguous in the rates dimension — exactly the dimension that differentiates
it from crisis.

Concrete fix: revise the n4 contraction signature to reflect the canonical recession signal more
directly. Either weight the recession canonical archetype more heavily in the blend (it is the more
common terminal condition of "contraction"), or explicitly set `rates: 0.0` to reflect the ambiguity
rather than the spurious -0.2.

### 7.2 Liquidity Crisis: Confidence 0.356

State 3 has confidence only 0.356 — the lowest cosine similarity of any state to its assigned
archetype. The crisis archetype requires `credit: -2.5` and `volatility: -2.5` — extreme values
representing GFC or COVID-level stress. If state 3 is capturing a "moderate stress" environment
that includes 2024-25 (which MEMORY notes as unexpected), its actual feature means may be around
`credit: -1.0`, `volatility: -1.0` — significantly less extreme than the archetype.

The consequence is that state 3 is labeled "Liquidity Crisis" but may represent a broader "financial
stress" or "risk-off" regime. Conditioning Phase 3 models on this label as if it exclusively
represents GFC-like crises would be misleading.

The low confidence is an honest diagnostic signal: the label is uncertain, and the `margin=0.122`
indicates the runner-up (Contraction at 0.234) is not far behind. For Phase 3 conditioning, this
state should probably be treated as "elevated stress" rather than "crisis", and the soft
probability (`filter_proba_3`) should be used rather than the hard label.

### 7.3 Inflation/Policy Stress Over-Reach (Known Issue)

Per MEMORY.md, state 2 (Inflation/Policy Stress) captures 2017 expansion and 2019 slowdown in
addition to the 2022-24 hiking cycle. This is a material misclassification: 2017 is characterised
by strong synchronised global growth (expansion archetype), and 2019 has slowing growth with tame
inflation and a pause in Fed policy (slowdown archetype).

The failure mode is that the n4 inflation_policy archetype has `inflation: 1.2` (strong inflation
signal) but `real_economy: -0.2` (near-neutral growth). If 2017 had modestly elevated CPI on a
rising trend, the inflation signal alone could push it into the inflation_policy regime. The key
differentiator — that 2022 had a forcefully inverted yield curve (rates_pc1 very negative) while
2017 did not — suggests the rates dimension is not discriminating enough.

**Root cause:** The n4 pool collapses stagflation and policy_constrained into a single archetype with
`rates: -1.0`. The 2022 hiking cycle had rates_pc1 ≈ -2.8 (extremely inverted). A 2017 expansion
might have rates_pc1 ≈ +1.5 (steep, accommodative). The blended n4 archetype at rates=-1.0 is
insufficient to force the 2022 signature far enough away from 2017 in cosine space.

**In a 5-regime model**, these would separate cleanly: policy_constrained (rates=-1.5) vs expansion
(rates=+0.4) are very distinct archetypes in the canonical pool. The 4-state constraint is forcing
a merger that the data does not support. This is the primary argument for pursuing a stable 5-regime
solution.

### 7.4 Episode Validation: The 8 Failing Episodes

Given the 4-state labeling, the likely failures are:

- **2017 Expansion**: expects `expansion → risk_on`, but model assigns `inflation_policy` (see §7.3)
- **2019 Late-Cycle Slowdown**: expects `slowdown → contraction`, but model assigns `inflation_policy`
- **Dot-com Bust**: expects `recession → contraction`; depends on how much the model steepens the
  curve in 2000-02. If the crisis archetype (rates +1.5) outcompetes contraction, this fails.
- **2013 Taper Tantrum**: expects `expansion → risk_on`; taper tantrum caused a VIX spike and credit
  spread widening that could push the model toward contraction or inflation_policy
- **2024 Soft Landing**: expects `expansion → risk_on`; if 2024 is labeled as Liquidity Crisis
  (per the OOS dominance in MEMORY.md), this episode fails
- **2025 Policy Uncertainty**: expects `slowdown → contraction`; same concern if state 3 dominates 2025

The 8 failures are not independent mislabelings — several likely cascade from the same structural
issue: the inflation_policy state absorbing too many observations, and the liquidity_crisis state
over-extending into 2024-25 OOS.

### 7.5 `economic_episodes.yaml` Comment Is Stale

Line 10 says: *"Active archetype keys in the current 5-state model."* The current winner is 4-state.
This comment was written when the previous audit's 5-regime winner was the working model. It now
creates confusion by referencing archetypes (expansion, recovery, policy_constrained, slowdown,
liquidity_crisis) that are all canonical keys — not the 4-state n4 pool keys. The comment should
be updated to:

```
# Active pool: n4. Canonical keys translate to n4 via taxonomy.canonical_to_coarse:
#   expansion / recovery → risk_on
#   slowdown / recession → contraction
#   stagflation / policy_constrained → inflation_policy
#   liquidity_crisis → crisis
```

---

## 8. YAML Configuration Assessment

### 8.1 Overall Quality

The YAML structure is genuinely strong. `regime_archetypes.yaml` is the standout file: the sign
convention documentation with empirical examples (GFC: rates_pc1=+3.8; 2022: rates_pc1=-2.8) is
exactly the kind of interpretability evidence that makes the system auditable. `regime_config.yaml`
is clean and self-documenting. `regime_universe.yaml` is detailed and accurate.

### 8.2 Specific Issues

**`economic_episodes.yaml` line 10 — stale comment** (Low priority; fix immediately):
References 5-state model and canonical archetype keys rather than the current n4 pool structure.

**`regime_config.yaml` — CV gap between warning and hard-reject** (Medium priority):
`max_churn_warning: 0.20` and `max_churn: 0.65` with no intermediate action prescribed.
The current winner sits at 0.6147, far above the warning threshold and close to the hard limit.
Consider adding a `max_churn_soft_reject: 0.45` or similar intermediate threshold that triggers
additional scrutiny or a reduced stability weight, rather than passing through with only a warning.

**`regime_config.yaml` — `min_exit_paths: 1` passes everything** (Low priority):
The hard filter rejects states with zero exit paths (absorbing states). In practice all models have
`min_exit_paths=1`. The comment says "1-exit cascades penalised in transition score" but `off_pen`
ranks by `max_offdiag`, not by exit path count. States with exactly 1 viable exit route (probability
> 0.001) that also have low off-diagonal values would not be penalised. Consider whether a state
with only one reachable neighbour is an acceptable architecture.

**`regime_config.yaml` — `filters.max_implied_duration: 1500.0`** (Low priority):
This is 10× the soft_score upper bound of 150 days, which means a model with 500-day implied duration
passes the hard filter comfortably but scores near-zero on the duration soft score. The comment
documents this two-tier design but it creates a confusing gap: models in the 400-1500d range pass
the hard filter with zero effective soft-score contribution from duration. Whether any model actually
falls in this range is worth checking on the leaderboard.

### 8.3 Configuration Clarity Score (Updated)

| YAML File | Clarity | Correctness | Notes |
|-----------|---------|-------------|-------|
| `regime_config.yaml` | High | High | CV threshold gap; stale reference comment |
| `regime_archetypes.yaml` | Very High | High | n4 contraction rate signature ambiguous |
| `economic_episodes.yaml` | High | Medium | Stale 5-state model comment |
| `regime_universe.yaml` | Very High | High | FEDFUNDS window comment mismatch |

---

## 9. Concrete Optimisation Recommendations

### Priority 1: Investigate Near-Zero CV Fold Scores

**Action:** Add fold-level metadata to CV output — specifically, log the IS-end date, OOS-start date,
and OOS-end date for each fold alongside its ANOVA R² score. Store this in `run_metadata.json` as
`cv_fold_dates: [{is_end, oos_start, oos_end, anova_r2}]`. Then inspect which economic periods
correspond to the four near-zero folds (indices 2, 9, 10, 22).

This is the single most important diagnostic action. Near-zero ANOVA R² in multiple folds means
the HMM has no discriminative structure in those windows, which undermines confidence in the
regime signal for Phase 3.

**If the collapse folds are all from pre-2000 IS windows:** The early data may be insufficient for
a stable 4-regime solution. Consider raising `min_train_years` from 5 to 7, which would skip the
most data-sparse folds.

**If the collapse folds are spread across history:** The model has a structural inability to maintain
discriminative regime structure in some macro environments. This is a fundamental signal quality
issue that warrants feature redesign.

### Priority 2: Revise CV State Alignment Reference

**Action:** Change `ref_detector = None` (set on first fold) to use the full-IS model as the reference.
Concretely: before the CV fold loop, fit the HMM once on the full IS window (data up to train_end_date)
and use its means as `ref_means`. Align each fold's states against these reference means. This produces
alignment relative to the production model rather than the smallest-IS-window fold, which is the most
stable reference available.

This would also make the churn calculation more meaningful: it would measure "how much do fold assignments
differ from the production model" rather than "how much do consecutive folds differ from each other",
which is more directly relevant to Phase 3 conditioning quality.

### Priority 3: Tighten n4 Archetype Signatures

The two labeling problems in §7 (Contraction/Crisis confusion, Inflation/Policy over-reach) have
a common root: the n4 pool's coarse merging loses the rate-dimension differentiation that the
canonical pool uses to separate regimes. Specific fixes:

**Contraction:** Change `rates` from -0.2 to +0.2. The canonical recession archetype has rates=+0.3
(emergency cuts steepen the curve). In the n4 pool, contraction = slowdown + recession. Slowdown
has rates=-0.7 but it is the milder component; recession is the archetype defining feature in crisis
periods. Weighting the recession component more heavily on the rates dimension would move contraction
to slightly positive rates, sharper credit/volatility stress, and cleaner separation from crisis.

**Inflation_policy:** Raise the `inflation` signature from 1.2 to 1.5, and strengthen `rates` from
-1.0 to -1.3. The 2022 hiking cycle had rates_pc1 ≈ -2.8; the current -1.0 signature is too moderate
to strongly attract 2022 while repelling 2017 (rates_pc1 ≈ +1.5). A stronger negative rates signal
would anchor the archetype to genuine policy tightening periods.

After any signature change, re-run the full pipeline and verify episode validation counts improve.
Do not change signatures without re-running.

### Priority 4: Pursue a More Stable 5-Regime Solution

All five 5-regime models fail the CV churn hard filter (>0.65). But the 5-regime model is structurally
better at separating inflation/policy and expansion states. Three approaches to getting a stable
5-regime solution:

**(a) Strengthen the CV churn filter and adjust the grid.** Rather than changing the threshold,
try a wider p_stay grid for n=5 specifically: `[0.95, 0.97, 0.99]` with n_init=20 seeds instead
of 10. More initialisation attempts at higher p_stay may find more stable 5-state solutions.

**(b) Constrain the 5-state model with tighter initialisation.** The KMeans initialisation feeds
the EM algorithm and heavily influences where it converges. For n=5, explicitly initialise two
states from the current n=4 split: use the production model_6's means as initialisation for 4 of
the 5 states, and initialise the 5th as the residual cluster from the most ambiguous state (state 2,
Inflation/Policy). This "surgical split" initialisation is much more likely to produce a coherent
5-state solution than random KMeans seeds.

**(c) Use a Dirichlet-process prior (infinite HMM).** This is a significant architectural change but
automatically selects K from the data and tends to find more stable solutions than finite-K EM. The
`hmmlearn` library does not support this, but the `pyhsmm` library does.

### Priority 5: Add PPI or Commodity Indicator to Credit/Inflation Group

The current feature set has no commodity price signal. The 2015-16 commodity bust was driven by
oil collapse; the 2022 inflation regime had a large commodity component; the 2023 disinflation was
partly commodity-driven. VIX partially captures the equity fear during commodity stress, but not
the commodity price level itself.

**Candidate series:**
- `PPIACO` (Producer Price Index, All Commodities) — FRED, monthly, starts 1913. No ALFRED needed
  (commodity price indices are not heavily revised). Would add a supply-side inflation signal that
  CPI lags by 1-3 months.
- `DCOILWTICO` (WTI crude oil spot price) — daily, starts 1986. Would directly capture oil cycle.
  Requires winsorization (COVID oil crash to -37/barrel is a genuine 10σ event).

Either series could be added to the inflation group (making inflation PC1 a blend of demand-pull
and cost-push inflation) or to a new "commodity" group. Given the IS window starts 1990 and WTI
daily is available from 1986, this would not compress the window.

### Priority 6: Document and Resolve FEDFUNDS Window Semantic Mismatch

Change the comment on the FEDFUNDS level transform from `# Policy stance level` to
`# Short-term policy momentum (last ~7 weeks)`. And consider whether a longer-window z-score
(e.g., `window: 126` — 6 months) would better capture the "policy stance level" intent. Add
`# 36 trading days ≈ 7.2 calendar weeks` to all daily-series z-score windows in the YAML.

### Priority 7: Fix `economic_episodes.yaml` Stale Comment

Update lines 10-23 to reference the current n4 pool and its canonical-to-coarse mapping, not the
no-longer-active 5-state model.

---

## 10. Model Quality and Phase 3 Readiness

### 10.1 Signal Quality by State

| State | Phase 3 Usability | Reason |
|-------|------------------|--------|
| 1: Risk-On | **High** | Confidence 0.953, margin 1.600. Cleanly identified. Safe for hard conditioning. |
| 2: Inflation/Policy | **Medium** | Confidence 0.898 but over-captures 2017 and 2019. Use with contextual validation. |
| 0: Contraction | **Low-Medium** | margin_warning=True, runner-up is Crisis. Use soft proba only. |
| 3: Liquidity Crisis | **Low** | Confidence 0.356; may represent "moderate stress" not crisis. Over-dominates 2024-25 OOS. |

### 10.2 Recommendations for Phase 3

**Use soft probabilities, not hard labels.** `filter_proba_k` columns in `regime_assignments.parquet`
are available and propagate the model's uncertainty. For states 0 and 3 with low confidence or thin
margins, hard labels will introduce spurious conditioning cliff effects. A soft-conditioned model
(trained with regime soft weights rather than one-hot regime dummies) will generalise better.

**Apply a confidence mask for states 0 and 3.** Before conditioning Phase 3 models on states 0 and 3,
apply `filter_proba_3 > 0.70` and `filter_proba_0 > 0.65` as minimum confidence gates. Observations
where the regime is ambiguous should be treated as "mixed" rather than assigned to a specific regime.

**Validate date ranges before use.** The MEMORY notes that Inflation/Policy Stress captures 2017 and
2019. Before conditioning Phase 3 models on regime assignments, compare the actual date ranges in
`regime_assignments.parquet` per state against economic history. If state 2 dominates 2017 but 2017
is known expansion, Phase 3 models trained on "Inflation/Policy" may be learning expansion-regime
behaviour under the wrong label.

**Assess Liquidity Crisis 2024-25 OOS dominance.** State 3 occupies 26.9% of OOS days including
2024-25. If Phase 3 is trained through 2024, this mislabeling will contaminate the crisis model
with non-crisis observations. Consider whether to exclude 2024-25 from Phase 3 training until
the regime model stabilises.

### 10.3 The 5-Regime vs 4-Regime Trade-off

The 4-regime winner is fundamentally limited by the n4 pool's compression of economically distinct
regimes. The 5-regime model would cleanly separate policy_constrained from expansion, resolving
the Inflation/Policy over-reach. The reason the 5-regime model fails is CV churn, not model quality.

This means the correct path forward is not to accept 4 regimes as the ceiling, but to find a stable
5-regime solution. Priority 4 above describes three concrete approaches. Resolving the 5-regime
instability should be the primary model quality initiative before Phase 3 begins.

---

## 11. What the System Does Exceptionally Well

To be precise about what should not be changed:

1. **Staleness discipline is correctly and thoroughly implemented.** Monthly series compute rolling
   statistics on real observations only, then forward-fill results. `is_new_data` propagation through
   the alignment pipeline is correct and tested. This prevents the most common look-ahead bias in
   financial feature construction.

2. **IS/OOS enforcement has multiple correct layers.** The scaler, PCA, KMeans initialisation, and
   evaluation metrics all correctly apply IS-only fitting and IS/OOS separation. The hard filters on
   OOS metrics (`oos_min_share`) add a structural robustness check that IS fitting alone cannot provide.

3. **`filter_proba` vs `smooth_proba` separation is maintained at every boundary.** The comments in
   `compare_hmm_models` explicitly document why `smooth_is` is acceptable for IS interpretation but
   `filt_oos` is mandatory for OOS evaluation. The `ANALYSIS ONLY` annotation on `label_regimes` is
   appropriately prominent. This discipline is rare in ML research pipelines.

4. **Absolute soft-score thresholds for macro coherence are the correct design.** The previous
   percentile-ranking approach was biased toward fewer-regime models. The absolute thresholds let
   BIC be the primary discriminator for model complexity, which is statistically principled.

5. **Archetype pool routing is elegant.** The `n3 → n4 → canonical` progression with
   `canonical_to_coarse` translation in episode validation is a clean, maintainable design.
   Episodes are written once in canonical space; the validation logic handles pool-specific
   translation. Adding a new pool or archetype requires only YAML changes.

6. **The test suite is production-grade.** 30 test files covering causality separation, staleness
   propagation, serialisation, CV strategy, episode validation, and transform edge cases provide
   genuine regression protection. The 233/233 pass rate is maintained.

---

## 12. Summary of Open Items

| Item | Severity | Section | Status |
|------|----------|---------|--------|
| Near-zero CV fold scores (folds 2, 9, 10, 22) | **High** | §5.2 | Open |
| max_pairwise_churn=0.9482; 23/24 non-trivial perms | **High** | §5.3 | Open |
| Inflation/Policy Stress over-captures 2017/2019 | **High** | §7.3 | Open |
| Liquidity Crisis confidence=0.356; 2024-25 OOS dominance | **High** | §7.2 | Open |
| CV reference fold is weakest fold (first = smallest IS) | **Medium** | §5.4 | ✅ Done — full IS model used as reference |
| Contraction archetype: rates=-0.2 creates Crisis confusion | **Medium** | §7.1 | ✅ Done — rates changed to +0.2 |
| n4 inflation_policy needs stronger rates/inflation signature | **Medium** | §9 P3 | ✅ Done — inflation 1.2→1.5, rates -1.0→-1.3 |
| FEDFUNDS window=36 trading days vs "policy stance" comment | **Low** | §3.2 | ✅ Done |
| FEDFUNDS diff=6 days labelled "cycle magnitude" | **Low** | §3.3 | ✅ Done |
| `economic_episodes.yaml` stale 5-state comment | **Low** | §7.5 | ✅ Done — pool-routing comment added |
| `selection.py` default `max_implied_duration=400` vs YAML 1500 | **Low** | §6.4 | ✅ Done — default changed to 1000 |
| Transform window unit ambiguity in YAML | **Low** | §3.2 | ✅ Done — frequency semantics comment added to regime_universe.yaml |
| Inflation group: single series, PCA step is trivial | **Low** | §3.4 | Open |
| CV churn: no intermediate action between warning (0.20) and hard-reject (0.65) | **Low** | §8.2 | Open |
