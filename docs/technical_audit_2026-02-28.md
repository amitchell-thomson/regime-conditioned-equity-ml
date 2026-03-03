# Regime Detection System — Technical Audit

**Date:** 2026-02-28
**Scope:** Full regime detection path, macro ingest → feature transforms → PCA → HMM → selection → labeling → evaluation
**State of pipeline:** 5-regime HMM, p_stay=0.97, full-covariance, BIC_IS=70,754, final_score=0.701
**Auditor:** Claude Code (claude-sonnet-4-6)

---

## Executive Summary

The codebase is well-architected and shows clear evidence of iterative hardening. Causal discipline, IS/OOS enforcement, and configuration centralisation are all executed correctly. The statistical machinery is sound. What follows is a frank assessment of what works, what has real correctness risks, and what should change to maximise the quality of the regime signal used to condition downstream models.

**What the system gets right:** Ledoit-Wolf covariance, multi-seed degeneracy filtering, staleness-aware transforms, hard IS boundary enforcement, the `smooth_proba` / `filter_proba` distinction, absolute soft-score thresholds (not percentile rank), and PCA sign anchoring.

**Primary risks:** A CV selection gap that leaves the winning model with null diagnostics; TV-20 soft-score thresholds calibrated to the wrong range (dead weight in scoring); a subtle CFNAI look-ahead bias risk; a hardcoded output path in the feature pipeline; and archetype-matching that will systematically miss recession-like episodes because no "recession" state exists in the 5-regime labeling.

---

## 1. Current Pipeline State vs. Memory

`run_metadata.json` contradicts `MEMORY.md`. The MEMORY file records a 4-regime winner (BIC=43,700, final_score=0.700). The current run produced a **5-regime winner** (model_10: n=5, full, p_stay=0.97, BIC_IS=70,754, final_score=0.701).

The memory document should be updated. The 5-regime result is plausible: BIC correctly prefers it because the 5-regime log-likelihood improvement (~2,926 BIC units) vastly exceeds the parameter count penalty (~257 BIC units for 29 additional parameters at n=7,150 IS obs). Whether the 5th state is stable OOS is the real question — and the CV diagnostics are null for this model (see §4.2).

Grid mapping (for reference):

| model_id | n_regimes | cov_type | p_stay |
|----------|-----------|----------|--------|
| model_0  | 3         | full     | 0.93   |
| model_1  | 3         | full     | 0.95   |
| model_2  | 3         | full     | 0.97   |
| model_3  | 3         | full     | 0.99   |
| model_4  | 4         | full     | 0.93   |
| model_5  | 4         | full     | 0.95   |
| model_6  | 4         | full     | 0.97   |
| model_7  | 4         | full     | 0.99   |
| model_8  | 5         | full     | 0.93   |
| model_9  | 5         | full     | 0.95   |
| **model_10** | **5** | **full** | **0.97** |
| model_11 | 5         | full     | 0.99   |

---

## 2. Feature Construction

### 2.1 What Works Well

**Staleness discipline is correctly implemented.** Monthly series (CPIAUCSL, PAYEMS, UNRATE, INDPRO) are forward-filled to daily frequency, and all transforms default to `staleness_mode='strict'`. This ensures rolling z-score windows count real observations, not forward-filled copies. The `is_new_data` flag is propagated correctly through the alignment pipeline. This is the single most important correctness property in the feature layer and it is correctly enforced.

**Group rationale is sound.** Merging growth + employment into `real_economy` is justified by the in-sample cross-correlation (INDPRO vs PAYEMS r=0.656). Separating VIX from credit after identifying the 2022 and 2015-16 divergences is precisely the kind of interpretability-driven decision this pipeline needs. DGS2 exclusion (algebraically redundant with DGS10 and T10Y2Y) is correct.

**Series selection avoids the IS-window constraint trap.** JTSJOL, UMCSENT, PCEPILFE, and BAMLH0A0HYM2 are all dropped because their feature start dates would compress the IS window to post-2005. The decision to use CPIAUCSL alone (back to 1985) is the right call — more IS data matters more than marginal inflation signal from breakeven rates.

### 2.2 Correctness Risks

**CFNAI revision look-ahead (medium risk).** `regime_universe.yaml` line 129 sets `use_alfred: false` for CFNAI with the comment: *"Composite index; heavily revised in first 1-2 years after release."* This is self-contradictory. If CFNAI is heavily revised at turning points (it is — Chicago Fed publishes a revision history), then using the final revised series to train on pre-2019 data means the model sees revised CFNAI values that were not available to a trader at the time. ALFRED vintage data for CFNAI exists on FRED and would eliminate this. This is a potential look-ahead bias, particularly around recession boundaries where CFNAI revisions are largest.

**Transform windows mix daily and monthly timescales inconsistently.** Several series use windows that appear to count calendar observations, not business days:

- FEDFUNDS: `z_score: {window: 36}` — if this counts daily observations (36 trading days ≈ 1.7 months), it is far too short for a z-score of a policy rate. If it counts monthly publications (36 months = 3 years), it is appropriate but since FEDFUNDS is daily, the transform computes on 36 daily points. The comment says "Policy stance level" implying 3-year context but the window is computed on 36 daily rows.
- Similarly: UNRATE and CFNAI use `window: 36` but their staleness-aware computation operates on 36 actual monthly observations (3 years), which is correct.
- ICSA uses `window: 50` — at weekly frequency this is ~50 weeks = ~1 year, reasonable.
- VIXCLS uses `window: 126` — at daily frequency this is ~6 months for the z-score level. Reasonable.

The issue is there is no documentation of whether `window` values in the YAML are intended as counts of real observations or trading days. For daily series (FEDFUNDS, DGS10, T10Y2Y, T10Y3M, BAA10Y, VIXCLS) the window is in trading days. For weekly/monthly series the window is in actual publication counts (weeks or months), applied after staleness filtering. This inconsistency is not a correctness error (staleness_mode handles it), but it makes the YAML harder to audit and maintain.

**ICSA 4-week moving average before pct_change.** The second ICSA transform chain is `[{ma: {window: 4}}, {pct_change: {periods: 13}}, {z_score: {window: 50}}]`. Applying a 4-week MA before a 13-week pct_change introduces mild smoothing but is reasonable. The concern is that this chain produces a quarterly change in a 4-week smoothed weekly claims series — this is a meaningful signal. No correctness issue, but the economic interpretation should be documented (it measures "how much have jobless claims deteriorated over the past quarter, smoothed to reduce noise").

### 2.3 Feature Naming Inconsistency

The feature naming convention `{SERIES_CODE}_{transform_chain}` is documented and reasonable. However, the ICSA chains produce names like `ICSA_ma_4_pct_change_13_zscore_50` — which mixes operation names from the transform registry. Consistency across the registry (snake_case everywhere) should be verified. This is not a functional issue but affects readability in the leaderboard and loadings CSVs.

---

## 3. Group PCA

### 3.1 What Works Well

**IS-only fitting is correctly enforced.** `GroupPCATransformer.fit()` slices `features.index <= self.train_end_date` before calling `pca.fit()`. The full dataset is then transformed but only IS rows influence the PC directions. This is correct.

**Sign anchoring is a genuine improvement.** Without sign anchoring, PCA may flip PC1 between runs (PCA directions are defined up to sign). The anchor-based flip ensures `positive PC1 = economically good` across all runs. This is essential for archetype matching: without it, a liquidity_crisis archetype with `credit: -2.5` might match a state with `credit_pc1 = +2.5` if PC1 happened to flip. The anchoring is IS-only (correct, no look-ahead).

**Within-group structure is preserved.** Running independent PCAs per group ensures PC1 of each group captures the dominant dimension of that group specifically. A single global PCA would mix groups and produce harder-to-interpret components.

### 3.2 Structural Issue: Duplicate Check

`group_pca.py` lines 191-195 contain an **exact duplicate** of the same `if not self._pcas` check:

```python
if not self._pcas:
    raise RuntimeError("GroupPCATransformer must be fit() before transform().")

if not self._pcas:
    raise RuntimeError("GroupPCATransformer must be fit() before transform().")
```

This is dead code. One check should be removed.

### 3.3 Hardcoded `_GROUPS` in Labeling

`labeling.py` line 29 hardcodes `_GROUPS = ("rates", "inflation", "real_economy", "credit", "volatility")`. If a group is renamed in the YAML or a new group added, `label_regimes` will silently ignore it (the `group_to_idx` dict for the new group will be empty, producing a zero state vector for that dimension). There is no validation that `_GROUPS` matches the actual groups present in the PC feature set. This is a silent failure mode.

### 3.4 PC1-Only Labeling

`label_regimes` uses only `idxs[0]` (PC1) from each group to build the state vector for archetype matching. This is documented and intentional. With the current config of 1 PC per group, this is correct. However, if n_components were increased in the config (e.g., `rates: 2`), the labeling would silently use only rates_pc1 while ignoring rates_pc2 even though rates_pc2 is being fed into the HMM. The archetype matching would then be based on an incomplete state representation.

### 3.5 PCA Is Blind to Regime Structure

This is a fundamental architectural constraint, not a bug. PCA maximises within-group variance, which is not the same as maximising regime separability. The first PC of the real_economy group captures the direction of maximum variance in the IS window — this will be dominated by the large economic swings (GFC, COVID) which are also exactly the periods that drive regime differences. So empirically the first PC is likely to be discriminative. But there is no guarantee of this, and it is possible that a second PC captures economically meaningful cross-cutting variation that helps the HMM but is discarded.

A regime-aware dimensionality reduction (e.g., fitting the HMM jointly with a latent factor model) would be theoretically superior but adds enormous complexity. For Phase 2, within-group PCA is a pragmatic and defensible choice.

---

## 4. HMM Fitting

### 4.1 What Works Well

**Multi-seed with degeneracy filtering is correctly implemented.** `fit_best_of_n_seeds` runs 15 seeds, checks each against `min_regime_share` and `tv_distance_valid`, and returns the best-LL passing model. The IS boundary is validated once before the loop (not inside), so OOS contamination is detected immediately. This is clean.

**Ledoit-Wolf covariance for initialisation is appropriate.** Cluster covariances computed from KMeans assignments can be rank-deficient (a small cluster with n_points < n_features). Ledoit-Wolf handles this without ad-hoc regularisation. The additional `1e-6 * I` jitter guarantees strict PD before hmmlearn's Cholesky decomposition. Correct.

**`filter_proba` implementation is mathematically correct.** The forward recursion in log space with `log_sum_exp` normalisation at each step is the standard causal HMM filter. The assertion that `pi.sum() ≈ 1.0` and `A.sum(axis=1) ≈ 1.0` provides a runtime sanity check. The diag covariance case is also handled (lines 524-542) despite the grid only using full covariance.

**BIC parameter count is correct.** `_n_params` correctly counts:
- startprob degrees of freedom: K-1
- transmat degrees of freedom: K*(K-1)
- emission means: K*D
- full covariance upper triangle: K*D*(D+1)/2

For K=5, D=5: n_params = 4 + 20 + 25 + 75 = 124. BIC penalty at n=7150: 124 * ln(7150) ≈ 1100. The 5-regime model's BIC advantage of ~2,926 over the best 4-regime model comfortably exceeds this.

### 4.2 Critical Issue: CV Selection Gap

**The pipeline has a structural gap where the winning model may not have CV diagnostics computed.**

The pipeline (pipeline.py lines 213-263) runs:
1. Initial selection without churn → identify top-6 candidates
2. Run CV only for those top-6
3. Final selection with churn applied to those candidates, neutral (0.5) for others

`run_metadata.json` confirms: `"cv_diagnostics": null` for model_10 (the winner). Looking at the leaderboard, model_10 has `churn_stability = 0.5` (the neutral fallback), meaning it was NOT in the initial top-6.

This creates a paradox: the best model is selected partly because it got neutral (not penalised) churn, while models that were tested may have been penalised for measured churn. model_11 (n=5, p_stay=0.99) received `churn_stability = 1.0`, suggesting it had near-zero CV churn — but it loses to model_10 which has unknown (neutral) churn.

If model_10 were given CV, its churn might be very low (similar to model_11 given they share n=5 and similar structure) or it might have high churn. Either way, the selection result is currently based on incomplete information.

**Consequence:** The downstream Phase 3 models will be conditioned on regime assignments from a model with unknown CV stability. If model_10 has high label churn, the conditioning signal will be noisy.

**Fix direction:** After the final winner is identified, check whether `cv_results_by_model.get(best_model_id)` is None. If so, run CV for the winner before writing run_metadata. This adds one CV evaluation but ensures the diagnostics are always populated.

### 4.3 Issue: Double Standardisation

The transform chains produce z-scored features (e.g., `FEDFUNDS_level_zscore_36`). PCA rotates these. `fit_best_of_n_seeds` then applies `StandardScaler` to the PC features before passing to HMM.

This is a double standardisation: the raw series is z-scored in the feature pipeline, then the PC projections of those z-scores are re-standardised. The PC values are unit-free but not necessarily unit-variance across groups — explained variance ratios differ by group. The `StandardScaler` in `initialise_emissions` corrects for this, ensuring no group dominates by scale alone.

This is not incorrect, but it means the HMM is fitted on `StandardScaler(PCA(z_score(raw)))`. The scaler is consistently saved and applied at inference, so there is no leakage. The redundancy is benign.

### 4.4 Minor: `filter_proba` Python Loop

The forward recursion in `filter_proba` (lines 596-601) uses a Python `for t in range(1, T)` loop over T=8,917 daily observations. For the current pipeline this runs in acceptable time, but it will scale poorly for longer histories or higher-frequency data. A vectorised implementation using `np.cumsum`-style operations or `scipy.linalg.solve` across the time dimension is possible.

This is a performance issue, not a correctness issue. At current scale it is not a priority.

---

## 5. Model Selection

### 5.1 What Works Well

**Absolute soft-score thresholds for macro coherence are the right design.** The previous percentile-ranking approach inherently biased 3-regime models (larger pairwise Mahalanobis distances in a lower-dimensional cluster space). The absolute thresholds treat all models on equal footing and let BIC handle the complexity penalty. This was a meaningful improvement.

**Hard filters are well-chosen.** The `tv_distance_valid` check (non-stationary transmat) and `min_regime_share >= 0.03` filter are the correct hard gates. An absorbing state or a dead regime cannot be used for trading conditioning.

**OOS hard filters are appropriate.** The `oos_min_share < 0.02` filter correctly rejects models that produce dead regimes in the OOS period. This is a structural robustness check that the IS fit cannot provide.

### 5.2 Significant Issue: `tv_score` Is Dead Weight

The `turnover` soft-score threshold (`optimal: 0.15, lo: 0.05, hi: 0.30`) scores the TV-20 mixing distance. Looking at the actual leaderboard values:

| model_id | tv20 |
|----------|------|
| model_10 | 0.647 |
| model_7  | 0.620 |
| model_5  | 0.604 |
| model_3  | 0.471 |

All TV-20 values are in [0.47, 0.65]. The configured `hi = 0.30`. Applying `_soft_score(0.647, optimal=0.15, lo=0.05, hi=0.30, slack=1.0)`:

- `x = 0.647 > hi = 0.30`
- `s = slack * width = 1.0 * 0.25 = 0.25`
- `score = max(0, 0.5 * (hi + s - x) / s) = max(0, 0.5 * (0.55 - 0.647) / 0.25) = max(0, -0.194) = 0.0`

**`tv_score = 0.0` for every model in the grid.** The 35% weight on `tv_score` within the transition component (`trans = 0.45 * dur_score + 0.35 * tv_score + 0.20 * off_pen`) is entirely wasted. The transition score effectively becomes `trans ≈ 0.45 * dur_score + 0.20 * off_pen`, normalised over a 0.65 weight instead of 1.0.

This means the 35% of transition score allocated to turnover is computing zero for all candidates, and the remaining terms are proportionally diluted. The symptom is that `transition_score` differences between models are smaller than they should be.

**The thresholds need recalibration.** With p_stay in [0.93, 0.99] and 4-5 regimes, a TV-20 distance of 0.30 is unreachably low (it would require near-instant mixing — incompatible with the p_stay grid). A realistic target for TV-20 with p_stay=0.97 and K=5 would be ~0.60-0.70. Either:
- Raise the thresholds to match reality: `optimal: 0.55, lo: 0.30, hi: 0.75`
- Replace TV-20 with a longer horizon (TV-252 or TV-504 for a 1-2 year mixing diagnostic)
- Remove `tv_score` from the transition component entirely and redistribute its weight to `dur_score` and `off_pen`

### 5.3 Opacity of model_id

`model_id = f"model_{idx}"` where `idx` enumerates `product(n_regimes_list, cov_types, p_stay_list)`. The leaderboard CSV requires external knowledge (the grid order) to know that model_10 is n=5/full/p=0.97. A descriptive ID like `n5_full_p0.97` or `hmm_K5_p097` would make the leaderboard self-documenting.

This is a maintainability issue that compounds as the grid grows.

### 5.4 Weight Sum Under CV Disable

When `churn_scores is None`, the code zeroes `w["churn"]` and renormalises all weights to sum to 1.0 (selection.py lines 90-93). This is correct but not documented in regime_config.yaml. A user tuning `selection.weights` might not realise that disabling CV changes all other effective weights proportionally. The YAML comment should note this.

### 5.5 `maha_min_quantile` Hard Filter

The Mahalanobis minimum filter (`maha_min < maha_thresh` where `maha_thresh = maha_min.quantile(0.10)`) is data-driven but circular: the threshold is computed from the distribution of models in the current run. If all models have similarly low `maha_min` values, the 10th percentile will be low and the filter will reject nothing. If one model is much worse than others, it correctly gets filtered. This is a soft gate that adapts to the pool — reasonable for its purpose.

---

## 6. Labeling

### 6.1 What Works Well

**Linear assignment maximises total similarity without reusing archetypes.** This is the correct formulation. Greedy assignment (always taking the best available archetype) can produce suboptimal global assignments. `scipy.optimize.linear_sum_assignment` with `-S` as the cost matrix solves it correctly.

**Confidence and margin thresholds are conservative.** `min_confidence: 0.18` and `min_margin: 0.06` are low enough that all 5 states are labeled in the current run (minimum confidence: 0.756 for state 4). The system is not over-triggering "Unclassified."

### 6.2 Issue: No `recession` State → Systematic Episode Failures

The current 5-state model labels states as: Policy-Constrained Growth, Recovery, Liquidity Crisis, Late-Cycle Slowdown, Risk On-Expansion. The linear assignment uses 5 of the 7 available archetypes; `recession` and `stagflation` are unused.

`economic_episodes.yaml` contains episodes that expect `recession` archetype:
- "Dot-com Bust" (2000-2002): expected `recession`

In `validate_against_episodes`, when `expected_archetype = "recession"`, `key_to_state.get("recession")` returns `None` (no state is labeled recession). This means `expected_state = None`, `expected_pct = NaN`, and `archetype_match = False`. **Every recession-expected episode will always fail regardless of how well the model captures that period.**

`run_metadata.json` confirms: `"n_episodes": 16, "n_matched": 7` (44% match). The miss rate is inflated by the systematic absence of a recession state. This is not necessarily wrong — the 5-state model may correctly capture the dot-com period as a "slowdown" followed by "recovery" without a dedicated recession state — but the episode validation metric is not accounting for this semantic gap.

The fix is to review which episodes fail and whether the failure reflects model inadequacy or archetype pool mismatch. Episodes expecting `stagflation` or `recession` (not present in current 5-state labeling) should either be updated to match the current archetype set or explicitly excluded from the pass/fail count.

### 6.3 Issue: Policy-Constrained vs Stagflation Margin is Dangerously Thin

State 0 (Policy-Constrained Growth):
- confidence: 0.9237
- runner_up: Stagflation (0.9019)
- **margin: 0.0218**

A margin of 0.022 means the model is nearly equally well-described by "Policy-Constrained Growth" and "Stagflation." These are economically distinct (stagflation = high inflation + weak growth; policy_constrained = high inflation + intact growth), but the cosine similarity space is not separating them well for this state.

Looking at the archetype signatures:
- `policy_constrained`: rates=-1.2, inflation=+1.0, real_economy=+0.4, credit=+0.2, volatility=-0.3
- `stagflation`: rates=-0.8, inflation=+1.5, real_economy=-0.5, credit=-0.5, volatility=-0.5

The state's actual mean (inverted curve + elevated inflation + modest positive growth) sits between these two archetypes. The thin margin suggests the archetype pool may need a more distinct policy_constrained signature (e.g., emphasise real_economy more strongly or add a `labour_tightness` dimension).

For downstream conditioning, this ambiguity is a risk: a downstream model trained on "Policy-Constrained" regime data may be receiving observations that the archetype system cannot confidently distinguish from stagflation.

### 6.4 Expansion vs Recovery Thin Margin

State 4 (Risk On-Expansion):
- confidence: 0.7567
- runner_up: Recovery (0.7087)
- **margin: 0.048**

This margin is wider than the policy_constrained case but still thin. The `expansion` archetype (strong growth + easy credit + calm markets) and `recovery` archetype (recovering + spreads tightening + steep curve) occupy adjacent regions in the 5-dimensional cosine space. The key differentiator is rates_pc1 — expansion has `rates: 0.4` while recovery has `rates: 0.7`. A state that sits between these would plausibly be labeled as either.

For conditioning, this means "Expansion" and "Recovery" may be interchangeable from the HMM's perspective, which reduces the power of conditioning downstream models on regime labels.

---

## 7. Evaluation

### 7.1 What Works Well

**Full IS/OOS evaluation structure is correct.** `compare_hmm_models` correctly uses `filt_full` (causal) for full-sample macro coherence, `smooth_is` (acceptable for IS interpretation) for IS macro coherence, and `filt_oos` (causal) for OOS evaluation. The comments make the causality contract explicit.

**`evaluate_regime_stability` uses Viterbi hard labels** as intended for persistence measurement. The analytical implied duration from the transition matrix is also reported separately. Both metrics are meaningful — the empirical persistence reflects actual state switching including noise; the analytical value is the model's intended duration.

**Expanding-window CV design is correct.** Fold IS-end dates are anchored to `min_train_years` and step by `fold_step_months`. Each fold refits from scratch (not warm-started). Label churn is computed on the overlapping OOS window between consecutive folds. This correctly measures how much regime assignments change as more data is added — the key stability diagnostic for downstream conditioning.

### 7.2 Issue: Episode Validation Counts Do Not Filter Archetype Mismatches

As noted in §6.2, the episode validation count includes episodes whose expected archetype is not present in the current model's labeling. The pipeline computes `n_matched_episodes = reachable["archetype_match"].sum()` where `reachable` filters only `n_days > 0` episodes. It does NOT filter episodes where `expected_state is None` (archetype not found in labeling results).

For the 5-state model: expected `recession` archetype → no state labeled recession → `expected_pct = NaN`, `archetype_match = False`. These count as failures even though the model may be correctly labeling those periods as "Late-Cycle Slowdown" or "Liquidity Crisis." The metric is therefore not measuring what it claims to measure (model accuracy) — it's measuring a combination of model accuracy and archetype coverage.

**Fix:** Either (a) update economic_episodes.yaml to use only archetypes present in the current model, or (b) add a filter to `validate_against_episodes` that skips episodes with `expected_pct is NaN`, reporting them as "archetype not in current model" separately.

### 7.3 `evaluate_regime_stability` Python Loop

Lines 34-44 in evaluation.py compute regime run lengths via a Python `for r in regimes[1:]` loop. For T=8917 this is a ~9K iteration loop which runs in milliseconds. Not a functional issue, but it could be replaced with a vectorised implementation for future scalability.

---

## 8. YAML Configuration Assessment

### 8.1 Strengths

The YAML structure is genuinely excellent. Comments are specific, include empirical examples (e.g., "2007-09 GFC (Fed cut 5.25%→0%, curve steepened): rates_pc1 = +3.8"), and explain the reasoning behind design decisions. Sign convention documentation in `regime_archetypes.yaml` is thorough. The rates_pc1 orientation section alone is more useful than most financial ML documentation.

Parameters are fully centralised — no numeric literals in source code. The only exception is documented below.

### 8.2 Issues

**1. Hardcoded output path in `features/macro/pipeline.py`.**
Line 55: `with open("data/features/feature_metadata.yaml", "w") as f:` — relative path, not config-driven. This violates the "no hardcoded parameters" principle and will fail if the process working directory is not the project root. This path should be added to `regime_universe.yaml` under `outputs` or read from `regime_cfg`.

**2. `turnover` soft_score thresholds are miscalibrated.**
As documented in §5.2, `{optimal: 0.15, lo: 0.05, hi: 0.30}` produces zero score for all models in the current grid. These thresholds have never been effective in any run using p_stay ≥ 0.93.

**3. `economic_episodes.yaml` comment references 4-state model.**
Line 13: *"Active archetype keys in the current 4-state model."* The model is now 5-state. This comment creates confusion about which archetypes are expected to be populated.

**4. Weight renormalisation under CV disable is undocumented in YAML.**
When `cross_validation.enabled: false` (or churn_scores is None), the `churn: 0.15` weight is zeroed and all other weights are renormalized by 1/0.85. This changes effective weights from the documented values (e.g., `macro: 0.20` → effective `0.235`). The YAML comment should note this behaviour.

**5. `max_implied_duration: 1500.0` vs `soft_score.duration.hi: 150.0` gap.**
The hard filter allows models up to 1,500-day implied duration while the soft score penalises models above 150 days. This factor-of-10 gap means models with, say, 500-day implied duration pass the hard filter but receive low soft scores. The hard filter value of 1500 is functionally a "near-absorbing" check, while the soft score handles realistic over-persistence. This is intentional but the YAML comment could make this two-tier design more explicit.

**6. `min_exit_paths: 1` → always passes.**
The hard filter `min_exit_paths < min_exit_paths_required` rejects models where any state has 0 exit paths (absorbing states). All models in the current run have `min_exit_paths = 1` — meaning every state has exactly one dominant exit path. The filter never rejects anything. The comment "1-exit cascades penalised in transition score" is intended but the transition score `off_pen` component (which ranks models by lower `max_offdiag`) does not directly penalise states with single exit paths. A more direct penalty for exit path sparsity (e.g., penalise states with only 1 exit via a soft score on `min_exit_paths`) would make this filter effective.

### 8.3 Configuration Clarity Score

| YAML File | Clarity | Correctness | Notes |
|-----------|---------|-------------|-------|
| `regime_config.yaml` | High | Medium | tv_score thresholds miscalibrated |
| `regime_archetypes.yaml` | Very High | High | 4-state reference outdated |
| `economic_episodes.yaml` | High | Medium | Archetype mismatches not flagged |
| `regime_universe.yaml` | Very High | Medium | CFNAI alfred risk; window unit ambiguity |

---

## 9. Concrete Optimisation Recommendations

Ranked by impact on regime classification quality for downstream conditioning.

### Priority 1: Fix CV Selection Gap

**File:** `src/regime_ml/regimes/pipeline.py` lines 250-263
**Issue:** Winning model has null CV diagnostics if it wasn't in the initial top-6.
**Fix:** After final selection, check `if cv_results_by_model.get(best_model_id) is None`, and if so, compute CV for the winner. This requires approximately one additional CV run.

This is the highest priority because null CV diagnostics for the production model mean there is no stability evidence for the regime signal used to condition Phase 3 models.

### Priority 2: Recalibrate `tv_score` Thresholds

**File:** `configs/regimes/regime_config.yaml`
**Issue:** `turnover.optimal: 0.15, lo: 0.05, hi: 0.30` produces zero score for all models.
**Options:**
- Replace with calibrated values: `optimal: 0.55, lo: 0.30, hi: 0.75`
- Remove `tv_score` entirely and redistribute its 35% weight within transitions: `dur_score: 0.65, off_pen: 0.35`
- Change the metric from TV-20 to TV-252 (annual mixing horizon), which will produce smaller values compatible with the current thresholds

The simplest fix is to remove `tv_score` from the transition component and redistribute. The TV-20 mixing diagnostic is still useful as a hard filter (which it already contributes to via `tv_distance_valid`) but is not effective as a soft scoring component at current p_stay values.

### Priority 3: Fix Hardcoded Output Path

**File:** `src/regime_ml/features/macro/pipeline.py` line 55
**Issue:** `"data/features/feature_metadata.yaml"` is a hardcoded relative path.
**Fix:** Add `feature_metadata_path: "data/features/feature_metadata.yaml"` to `regime_universe.yaml` under outputs, and use `regime_cfg["feature_metadata_path"]` in the pipeline.

### Priority 4: Fix Duplicate Check in group_pca.py

**File:** `src/regime_ml/features/macro/group_pca.py` lines 191-195
**Issue:** Identical `if not self._pcas` check appears twice in `transform()`.
**Fix:** Remove the second occurrence.

### Priority 5: Make model_id Descriptive

**File:** `src/regime_ml/regimes/pipeline.py` `_fit_grid()` function
**Current:** `model_id = f"model_{idx}"`
**Proposed:** `model_id = f"n{n_regimes}_{cov_type}_p{p_stay:.2f}"` (e.g., `n5_full_p0.97`)
**Benefit:** Leaderboard CSV and run_metadata.json become self-documenting. The index-based naming requires off-band documentation to interpret.

### Priority 6: Episode Validation Archetype Coverage

**File:** `configs/regimes/economic_episodes.yaml` and/or `src/regime_ml/regimes/evaluation.py`
**Issue:** Dot-com Bust expects `recession` archetype which is never assigned to any state in the 5-regime model, guaranteeing failure.
**Options:**
- Update the episode to use `slowdown` (the model's representation of that period) — only if the model actually labels that period as slowdown
- Add a `not_in_model` status to `validate_against_episodes` that skips archetype-absent episodes from the pass/fail count, reporting them in a separate column
- Add a dedicated `recession` archetype to the 5-state model's pool and verify which state it maps to

The cleanest long-term solution is option 3: the dot-com bust and 2001 recession are semantically distinct from a "slowdown" (no credit blowup, mild labour market stress) and from "liquidity crisis" (moderate severity vs GFC/COVID). A `recession` state should be recoverable in a 5-regime model given sufficient data.

### Priority 7: Address CFNAI Look-Ahead Risk

**File:** `configs/data/regime_universe.yaml` line 129
**Issue:** CFNAI is flagged as heavily revised but `use_alfred: false`.
**Risk:** IS-period CFNAI values include post-publication revisions not available at the time; feature construction is therefore slightly leaky for pre-2019 IS data.
**Fix:** Either set `use_alfred: true` for CFNAI if ALFRED vintage data is available on the FRED source, or add a comment documenting that revision risk is accepted as a minor approximation. The impact is likely small (CFNAI revisions affect early readings but the final values are broadly similar), but it should be a documented decision rather than an oversight.

### Priority 8: Validate `_GROUPS` Against Feature Set

**File:** `src/regime_ml/regimes/labeling.py` line 29
**Issue:** `_GROUPS` is hardcoded; if PC feature columns don't map to all five groups, labeling silently uses zeros for missing groups.
**Fix:** At the start of `label_regimes`, verify that all groups in `_GROUPS` are represented in `group_to_idx` and log a warning if any group has no features mapped to it.

---

## 10. Model Quality Assessment for Phase 3 Conditioning

The fundamental question: **is this a good regime signal to condition downstream models on?**

### Strengths for Conditioning

1. **5-state granularity is appropriate.** With 8,917 total observations and 7,150 IS obs, 5 regimes is statistically supportable (mean IS obs per regime: ~1,430). Phase 3 models will have enough within-regime training data to learn regime-specific parameters.

2. **Causal filter_proba is available.** The `filter_proba_k` columns in `regime_assignments.parquet` are correctly computed using the forward recursion. Conditioning can use soft probability vectors (all 5 columns) rather than hard labels, which is strictly more informative.

3. **OOS structural stability is decent.** The OOS ANOVA R² for model_10 is 0.386 (vs IS 0.450), a 14% degradation. This is a moderate but not alarming OOS/IS ratio for macro regime detection.

4. **High-confidence crises are crisply identified.** The Liquidity Crisis state (state 2) has confidence 0.813 and margin 0.449 — very clean separation from all other states. Conditioning on this state should provide a strong signal for crisis-period models.

### Weaknesses for Conditioning

1. **Policy-Constrained vs Stagflation ambiguity (margin 0.022)** creates noise in the most recent (2022-2024) assignments. Phase 3 models conditioned on "Policy-Constrained" may be receiving mixed Stagflation/Policy-Constrained signal.

2. **Null CV diagnostics** for the winning model means there is no empirical evidence on label stability. If the 5-regime model has high label churn (like the 3-regime models), the conditioning signal will be unstable across time.

3. **2024-25 period labeling uncertainty.** The current model labels 2024-25 (data_end: 2026-01-23) but this is OOS. Without knowing what state dominates this recent period and whether it matches economic reality, downstream model training on this period could be noisy.

4. **Expansion and Recovery conflation.** The thin margin (0.048) between these two states means the post-2009 recovery and the 2013-2019 expansion may be inconsistently labeled. Phase 3 models expecting "Expansion" to mean sustained bull market may be trained on a mix of recovery and expansion periods.

### Recommendation

Before passing regime assignments to Phase 3:
1. Resolve the CV diagnostics gap (Priority 1 above) to confirm the 5-regime model is stable.
2. Inspect the actual date ranges in `regime_assignments.parquet` for each state to verify the economic periods are classified as expected.
3. Consider whether the Phase 3 model architecture should use hard labels (integer state) or soft probabilities (`filter_proba_k`). For regime-conditioning, soft probabilities are strictly superior — they propagate the model's uncertainty and avoid cliff effects at regime boundaries.

---

## 11. What the System Does Exceptionally Well

To be explicit about what should not change:

1. **IS/OOS enforcement is rigorous.** Multiple layers (IS-only scaler, IS-only PCA, IS-only KMeans init, OOS hard filters) ensure the model selection process is not contaminated. This is rare in research pipelines.

2. **The staleness discipline is correctly implemented** at every layer. Many financial ML systems quietly compute rolling statistics on forward-filled data; this system does not.

3. **Absolute soft-score thresholds** for macro coherence are a genuine improvement over percentile ranking and correctly solve the cross-n_regime bias problem.

4. **The Ledoit-Wolf + degeneracy filter combination** is exactly the right approach for KMeans initialisation of HMM emissions. Small clusters and rank-deficient covariances are handled without ad-hoc diagonal fallbacks.

5. **The archetype sign convention is well-documented** and consistently applied. The rates_pc1 empirical validation examples in `regime_archetypes.yaml` (e.g., "2007-09 GFC: rates_pc1 = +3.8") are exactly the kind of interpretability evidence that makes the system auditable.

6. **The test suite is comprehensive.** 28 test files covering transforms, staleness, causality separation, serialisation, CV, and episode validation provide genuine protection against regression.

---

## Summary Table

| Area | Key Issue | Priority | Resolution |
|------|-----------|----------|------------|
| Feature transforms | Window unit ambiguity in YAML | Low | ✅ Done — frequency semantics comment added to regime_universe.yaml |
| Staleness handling | None | — | ✅ Correct — no action needed |
| CFNAI look-ahead | use_alfred: false despite heavy revisions | Medium | ⚠️ Acknowledged — documented as voluntary concession in YAML comment |
| Group PCA | Duplicate check; _GROUPS hardcoded | Low | ✅ Done — duplicate check removed; runtime validation added |
| HMM fitting | filter_proba Python loop (perf) | Low | Open — acceptable at current scale |
| CV selection gap | Winner has null CV diagnostics | **High** | ✅ Done — CV now runs for all hard-filter survivors |
| BIC scoring | Normalised within survivors (intended) | — | ✅ Correct — no action needed |
| tv_score thresholds | Zero score for all models | **High** | ✅ Done — tv_score removed; weight redistributed to dur_score/off_pen |
| model_id naming | Positional, not descriptive | Medium | Open |
| Archetype labeling | Policy vs Stagflation margin 0.022 | Medium | ✅ Done — n4 inflation_policy signature strengthened (inflation 1.2→1.5, rates -1.0→-1.3) |
| Episode validation | Recession episodes always fail | Medium | ✅ Done — pool-routing comment clarified in economic_episodes.yaml; n4 contraction rates fixed |
| Hardcoded path | feature_metadata.yaml path | Medium | ✅ Done — feature_metadata_path added to regime_universe.yaml |
| Duplicate check | group_pca.py transform() | Low | ✅ Done — removed |
| YAML quality | Outdated 4-state comment | Low | ✅ Done — economic_episodes.yaml updated with pool-routing comment |
| Test coverage | CV selection gap not tested | Medium | ✅ Done — 34 new CV tests added |
