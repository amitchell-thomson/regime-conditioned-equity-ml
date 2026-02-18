# TECHNICAL AUDIT: Regime-Conditioned Equity ML System

**Classification**: Internal Code Review Memo
**Date**: 2026-02-17
**Scope**: Pre-deployment quant hardening review, Phases 1-2

---

## 1. Macro Data Engineering & Integrity

### 1.1 CRITICAL: Staleness Detection via Value Equality Is Wrong

**File**: `src/regime_ml/data/macro/alignment.py:23-27`
```python
df["is_new_data"] = (
    df.groupby("series_code")["value"]
    .transform(lambda x: x != x.shift(1))
    .fillna(True)
)
```

**Why dangerous**: This detects "new data" by checking if the value *changed*, not whether a new observation was published. If CFNAI publishes the same value two months in a row (e.g., 0.00 → 0.00), this flags the second observation as `is_new_data=False`. The downstream staleness-aware transforms then skip that actual observation. Conversely, if FRED retroactively revises a value, the revised row gets flagged as "new data" even though no new economic observation occurred — it's a revision of an old one.

**Impact**: All staleness-aware transforms (`staleness_mode='strict'`) consume incorrect observation counts. Monthly z-scores computed over "36 actual observations" may actually be using 30 or 40 depending on how often values happen to repeat.

**Fix**: Track staleness by comparing against the known publication calendar for each series, or simply mark `is_new_data=True` at the source load stage, before forward-filling, by flagging rows that existed in the original data. Do not infer freshness from value changes.

**Priority**: **HIGH**

---

### 1.2 CRITICAL: No Macro Revision / Release Lag Handling

**File**: `src/regime_ml/data/macro/loaders.py` (entire module), `configs/data/regime_universe.yaml`

**Why dangerous**: FRED stores the *latest revised* values for all historical dates. Many macro series used here are subject to significant revisions:
- **CFNAI**: Revised monthly with 1-month lag, then benchmark-revised annually
- **INDPRO**: Preliminary → 1st revision → 2nd revision → benchmark revision
- **PCEPILFE**: Released with ~1-month lag, revised for 2+ months

The pipeline loads the current snapshot of FRED data and treats every value as if it were known at time `t`. This creates **lookahead bias**: the model trains on revised values that were not available at the time. For example, INDPRO for January 2020 may have been revised 3 times by March 2020, but the model assumes the final (revised) value was known on the January release date.

**Fix**:
1. Use ALFRED (Archival FRED) real-time vintage data, which stores each value as of its initial release date and subsequent revisions.
2. At minimum, add a `release_lag_days` field per series in `regime_universe.yaml` and shift dates forward by the publication lag. E.g., monthly CFNAI released end of following month: shift by ~30 business days.

**Priority**: **HIGH**

---

### 1.3 HIGH: StandardScaler Fitted on Full Training Set Before Split

**File**: `src/regime_ml/regimes/hmm.py:43-45`
```python
scaler = StandardScaler()
if scale_features:
    X_train_scaled = scaler.fit_transform(X_train)
```

**File**: `src/regime_ml/regimes/evaluation.py:268-269`
```python
X_full = df_selected.values
X_full_scaled = scaler.transform(X_full)
```

**Why dangerous**: The scaler is fitted on `X_train` (which in `compare_hmm_models` is implicitly the full sample since `initialise_emissions` was called earlier on the training data). However, in `compare_hmm_models`, the same scaler is used to transform the full dataset including OOS. If the scaler was fit on only the IS period, this is correct. But the flow is opaque — the scaler lives inside `model_data["scaler"]`, and there's no enforcement that it was fitted exclusively on IS data.

Additionally, `initialise_emissions` receives `df_train` but there's no enforcement that this is the IS subset. If the notebook passes the full dataset, the scaler sees OOS data.

**Fix**: Enforce IS-only scaler fitting. Add an assertion or wrapper that takes an explicit train/test split date and ensures the scaler never sees OOS data. Log the scaler's training period in model metadata.

**Priority**: **HIGH**

---

### 1.4 HIGH: `smooth_proba` Used for OOS Evaluation Leaks Information

**File**: `src/regime_ml/regimes/evaluation.py:288-289`
```python
smooth_is = model.smooth_proba(X_is) if X_is.shape[0] > 0 else None
smooth_oos = model.smooth_proba(X_oos) if X_oos.shape[0] > 0 else None
```

**Why dangerous**: `smooth_proba` runs the forward-backward algorithm within each slice independently, so IS and OOS are separated. However, using smoothed probabilities for OOS evaluation is still methodologically suspect: in a real trading system, you would only have `filter_proba` (causal). Evaluating with smoothed probabilities inflates OOS coherence metrics because the backward pass uses "future" information within the OOS window itself.

**Fix**: OOS evaluation should use `filter_proba` exclusively. Reserve `smooth_proba` for IS analysis/interpretation only.

**Priority**: **HIGH**

---

### 1.5 MEDIUM: Forward-Fill Before Trim Creates Silent NaN Propagation

**File**: `src/regime_ml/data/macro/alignment.py:72`
```python
aligned['value'] = aligned['value'].ffill()
```

**File**: `src/regime_ml/data/macro/cleaners.py:62-87` (`trim_to_common_start`)

The pipeline forward-fills values, then trims to the latest series start date. But if a series has a gap in the middle (e.g., FRED discontinues publication for a period), `ffill()` will silently propagate stale values indefinitely. There is no maximum staleness limit.

**Fix**: Add a `max_staleness_days` parameter per frequency. If a series hasn't updated in >2x its native frequency (e.g., 60 days for monthly), mark as NaN rather than forward-filling.

**Priority**: **MEDIUM**

---

### 1.6 MEDIUM: `days_since_update` Computed via Inefficient Loop with No Boundary Check

**File**: `src/regime_ml/data/macro/alignment.py:83-87`
```python
last_update_date = aligned[aligned['is_new_data'] == True].index
aligned['days_since_update'] = 0
for date in aligned.index:
    days_since = (date - last_update_date[last_update_date <= date].max()).days
    aligned.loc[date, 'days_since_update'] = days_since
```

**Why dangerous**: If there are no `is_new_data == True` entries before a given date (e.g., the series starts mid-way through the calendar), `last_update_date[last_update_date <= date]` is empty and `.max()` raises or returns `NaT`, causing the `.days` call to fail silently or produce incorrect values. Additionally, this is O(T * N) per series — extremely slow.

**Fix**: Use vectorized forward-fill of the last update date:
```python
aligned['last_update'] = aligned.index.where(aligned['is_new_data'] == True)
aligned['last_update'] = aligned['last_update'].ffill()
aligned['days_since_update'] = (aligned.index - aligned['last_update']).dt.days
```

**Priority**: **MEDIUM**

---

### 1.7 MEDIUM: Economic Redundancy in Feature Set

**File**: `src/regime_ml/features/macro/selection.py:36-71`

The ranked feature list includes multiple correlated signals:
- `DGS10_level_zscore_252` and `DGS2_level_zscore_252` — both nominal rate levels, highly correlated
- `T10Y3M_level_zscore_252`, `T10Y3M_diff_21_zscore_252`, `T10Y3M_diff_5_zscore_126` — three curve features

The feature validator checks 0.70 correlation, but the feature *selection* is hardcoded by economic intuition without empirical deduplication. In practice, DGS2 and DGS10 will have correlations >0.90 in many regimes.

**Fix**: Add a post-selection correlation check that either drops or PCA-combines features exceeding a threshold (e.g., 0.85). Or use the correlation matrix to inform the ranked list rather than relying purely on judgment.

**Priority**: **MEDIUM**

---

### 1.8 LOW: Hardcoded Absolute Path in Loader Default

**File**: `src/regime_ml/data/macro/loaders.py:12`
```python
source_dir: Union[str, Path] = "/Users/alecmitchell-thomson/Desktop/Coding/quant-data/macro",
```

**File**: `configs/data/regime_universe.yaml:5`
```yaml
data_path: "/Users/alecmitchell-thomson/Desktop/Coding/quant-data/macro"
```

**Why dangerous**: Machine-specific paths break portability. Anyone cloning this repo will get `FileNotFoundError` immediately.

**Fix**: Use environment variables or a `.env`-based path resolution. The loader should raise a clear error if `DATA_DIR` is not set rather than falling back to a hardcoded path.

**Priority**: **LOW** (but **HIGH** if publishing to GitHub)

---

## 2. Regime Model Specification (HMM / Switching Models)

### 2.1 HIGH: Gaussian Emission Assumption Not Validated

**File**: `src/regime_ml/regimes/hmm.py:187-194`
```python
self.model = hmm.GaussianHMM(
    n_components=n_regimes,
    covariance_type=covariance_type,
    ...
)
```

The HMM assumes Gaussian emissions. Macro z-scores may be approximately Gaussian in normal times, but financial stress indicators (VIX, NFCI) have heavy tails and skewness — precisely during regime transitions, which is when correct classification matters most.

**Problems**:
- VIX has strong right skew even after z-scoring (the z-score normalizes location/scale but not shape)
- During crises, observations may be 4-8 sigma events under Gaussian assumptions, causing the HMM to assign near-zero emission likelihood to the "correct" crisis regime
- Gaussian tails fall off as exp(-x²), real financial data falls off as x^(-α) — this mismatch means the model systematically underweights tail events

**Fix**:
1. Test with Student-t emissions (hmmlearn supports custom emission models via `_BaseHMM` subclassing)
2. At minimum, add QQ-plots per regime to validate Gaussian assumption
3. Consider winsorizing at 4σ before fitting to reduce tail sensitivity

**Priority**: **HIGH**

---

### 2.2 HIGH: No Convergence Diagnostics

**File**: `src/regime_ml/regimes/hmm.py:223-226`
```python
self.model.fit(X, **kwargs)
self.is_fitted = True
return self
```

The EM algorithm may:
- Not converge within `n_iter=1000`
- Converge to a local optimum
- Converge to a degenerate solution where one regime captures <1% of data

None of these are detected or logged. `hmmlearn` sets `model.monitor_.converged` but this is never checked.

**Fix**:
```python
self.model.fit(X, **kwargs)
if not self.model.monitor_.converged:
    warnings.warn(f"HMM did not converge after {self.n_iter} iterations")
self.is_fitted = True
```
Also log `self.model.monitor_.history` (the log-likelihood trace) to detect oscillation vs. convergence.

**Priority**: **HIGH**

---

### 2.3 HIGH: Single Random Seed, No Multi-Start

**File**: `src/regime_ml/regimes/hmm.py:174`
```python
random_state: int = 42,
```

EM for HMMs is highly sensitive to initialization. Using a single seed (42) means:
- You have no idea if this is a local or global optimum
- The solution may be unstable — a different seed could produce entirely different regimes
- The KMeans initialization helps, but KMeans itself is also sensitive to initialization

**Fix**: Run N initializations (e.g., 10-20 seeds), keep the model with highest log-likelihood that passes degeneracy filters. This is standard practice. Log all runs for reproducibility.

**Priority**: **HIGH**

---

### 2.4 MEDIUM: Degenerate Covariance Regularization Is Ad-Hoc

**File**: `src/regime_ml/regimes/hmm.py:68-77`
```python
if len(cluster_points) < 2:
    cov = np.eye(n_features) * 1e-6
else:
    cov = np.cov(cluster_points.T, ddof=1)
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-6)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
```

The `1e-6` floor on eigenvalues is arbitrary. If a cluster has only 3 points in 5 dimensions, the covariance will be rank-deficient. The eigenvalue floor makes it technically positive definite but the resulting Gaussian has extremely concentrated probability mass in the degenerate directions, causing numerical issues during EM.

**Fix**: Use Ledoit-Wolf shrinkage estimation for cluster covariances (already imported for evaluation in `evaluation.py`). Or set a minimum eigenvalue floor proportional to the data scale (e.g., 0.01 * median eigenvalue across all clusters).

**Priority**: **MEDIUM**

---

### 2.5 MEDIUM: `filter_proba` Cholesky Decomposition Can Fail

**File**: `src/regime_ml/regimes/hmm.py:295`
```python
L = np.linalg.cholesky(C)
```

If the covariance matrix `C` is not strictly positive definite (e.g., numerical precision issues after EM), `cholesky` will raise `LinAlgError`. There's no try/catch or jitter.

**Fix**: Add a small jitter: `C = C + np.eye(d) * 1e-8` before Cholesky. Or use `scipy.linalg.cho_factor` which handles near-singular cases more gracefully.

**Priority**: **MEDIUM**

---

### 2.6 MEDIUM: No State Permutation / Label Alignment

When fitting HMMs with different seeds or on different subsamples, state indices are arbitrary (state 0 in run A may correspond to state 2 in run B). There is no label alignment mechanism.

**Impact**: Comparing OOS regime shares to IS regime shares (as done in `select_best_hmm_model`) assumes state indices are consistent between IS and OOS. Since the same fitted model is used for both, this is fine within a single model. But if you ever compare two different fitted models, or extend to expanding-window refit, you'll need permutation alignment.

**Fix**: Implement Hungarian algorithm (scipy.optimize.linear_sum_assignment) for regime label alignment based on KL divergence between emission distributions.

**Priority**: **MEDIUM**

---

### 2.7 LOW: `n_iter=1000` May Be Excessive or Insufficient

**File**: `src/regime_ml/regimes/hmm.py:145`

1000 EM iterations is generous. Most well-initialized HMMs converge in 50-200 iterations. However, with poor initialization or near-degenerate solutions, 1000 may still not be enough.

**Fix**: Use `tol` (convergence threshold on log-likelihood improvement) instead of fixed iterations. hmmlearn supports `tol` parameter. Set `n_iter=500, tol=1e-4` and check convergence.

**Priority**: **LOW**

---

## 3. Model Selection Methodology

### 3.1 HIGH: No AIC/BIC Computed Anywhere

The model selection pipeline (`selection.py`) uses custom scoring but never computes AIC or BIC. For HMMs, the number of free parameters grows quadratically with `n_regimes` (transition matrix: K²-K, means: K*d, covariance: K*d*(d+1)/2 for full). Without a complexity penalty, the scoring system will systematically prefer higher-K models that overfit.

**File**: `src/regime_ml/regimes/selection.py` (absence)

**Fix**: Compute BIC = -2*LL + k*log(T) where k is the number of free parameters. Use it as either a hard filter or a scoring component. This is the single most important missing metric.

**Priority**: **HIGH**

---

### 3.2 HIGH: OOS Weight of 5% Is Negligibly Small

**File**: `src/regime_ml/regimes/selection.py:147-152`
```python
survivors["final_score"] = (
    0.40 * survivors["macro_score"] +
    0.30 * survivors["transition_score"] +
    0.25 * survivors["stability_score"] +
    0.05 * survivors["oos_macro_score"]
)
```

OOS performance gets 5% weight. This is essentially decorative. A model that collapses OOS but scores well IS will still win. This defeats the entire purpose of having an OOS evaluation.

**Fix**: Increase OOS weight to at least 20-30%. Better yet, restructure as: IS score selects candidates, OOS score selects the winner. A model that doesn't generalize OOS is worthless regardless of IS metrics.

**Priority**: **HIGH**

---

### 3.3 HIGH: Feature Selection Is Hardcoded, Not Part of Model Selection

**File**: `src/regime_ml/features/macro/selection.py:19-71`

`get_top_features()` returns a fixed, manually-ranked list. The `compare_hmm_models` function calls `get_top_features(n=n_features)` to select features. This means:
- Feature selection is not cross-validated
- The feature ranking was determined by the researcher looking at (presumably) in-sample results
- There's no way to know if the ranking holds OOS

**Fix**: Feature selection should be part of the model selection grid. Enumerate feature subsets (or at least different `n` values) as model configurations, and let the scoring function pick the best combination.

**Priority**: **HIGH**

---

### 3.4 MEDIUM: Hard Filter Thresholds Are Not Justified

**File**: `src/regime_ml/regimes/selection.py:34-40`
```python
min_share: float = 0.03,
max_share: float = 0.80,
oos_min_share: float = 0.02,
oos_max_share: float = 0.85,
max_implied_duration: float = 3000.0,
```

These thresholds are arbitrary magic numbers. Why is 3% the minimum share? Why 3000 days for max duration? A senior quant would immediately ask for the justification.

**Fix**: Either derive thresholds from economic priors (e.g., "we expect no regime shorter than 6 months ≈ 125 days, so minimum share = 125/5000 ≈ 2.5%") or use data-driven thresholds (e.g., percentiles of the candidate pool). Document the reasoning.

**Priority**: **MEDIUM**

---

### 3.5 MEDIUM: Rank-Based Scoring Loses Scale Information

**File**: `src/regime_ml/regimes/selection.py:117-118`
```python
def rrank(s, ascending=False):
    return s.rank(pct=True, ascending=ascending)
```

Rank-based scoring (percentile ranks) treats models that score 0.95 and 0.50 on Mahalanobis distance the same as models that score 0.51 and 0.50. If one model is dramatically better than all others on macro coherence, rank scoring won't reward it.

**Fix**: Use z-score normalization or min-max scaling instead of ranks for continuous metrics. Ranks are appropriate when the metric distributions are unknown, but here the metrics are well-defined.

**Priority**: **MEDIUM**

---

### 3.6 MEDIUM: No Cross-Validation or Expanding Window

The IS/OOS split is a single time-series split. This means:
- The split date is a free parameter (data-snooping risk if tuned)
- A single OOS window may be unrepresentative (e.g., if OOS is 2020-2026, dominated by COVID)
- No estimate of model stability across time

**Fix**: Implement expanding-window or rolling-window cross-validation. Refit HMM on {2005-2010, 2005-2012, 2005-2014, ...} and evaluate on the subsequent 2-3 year window. Score by average OOS performance across folds.

**Priority**: **MEDIUM**

---

## 4. Regime Interpretation & Economic Validity

### 4.1 HIGH: Label Set Is Hardcoded to Exactly 4 Labels

**File**: `src/regime_ml/regimes/labeling.py:82-87`
```python
labels = [
    ("Risk On - Expansion               ",                    +1.3*zg + 1.1*zl - 1.4*zi - 1.0*zs - 0.3*zr),
    ("Risk On - Stagflation",                                 -1.1*zg + 1.5*zi + 0.6*zs + 0.3*zr),
    ("Risk On - Policy-Contstrained Expansion",               +1.2*zi + 1.3*zr - 0.9*zl - 0.4*zg),
    ("Risk Off - Recession",                                  -1.4*zg - 0.6*zl + 1.4*zs + 0.3*zi),
]
```

Problems:
1. **Exactly 4 labels for any K**: If `n_regimes=3` or `n_regimes=5`, the labeling system is broken — multiple regimes will get the same label
2. **Typo**: "Policy-Contstrained" (missing 'r')
3. **Trailing whitespace** in "Risk On - Expansion               " — will cause display/comparison issues
4. **Label coefficients are arbitrary**: The weights (1.3, 1.1, -1.4, etc.) are not derived from data or economic theory. They're hand-tuned to produce "reasonable" labels.
5. **First label says "Risk On" but the model might assign it to a risk-off cluster** — the labeling is a post-hoc interpretation that may not match the statistical structure

**Fix**:
- Make the label set dynamic based on `n_regimes`
- Derive labels from data (e.g., which macro group has the largest absolute z-score defines the label)
- Remove magic coefficients
- Fix the typo and whitespace

**Priority**: **HIGH**

---

### 4.2 MEDIUM: No Validation Against Known Economic Episodes

There's no check that regimes align with known episodes. A credible system would verify:
- 2008 GFC → stress/recession regime
- 2020 COVID → brief stress → rapid recovery
- 2022 rate-hiking → tightening/inflation regime
- 2003-2007 → expansion regime

This is conspicuously absent from evaluation metrics.

**Fix**: Add a `validate_against_episodes()` function that checks regime classifications against a list of known economic episodes with expected regime types. This is not a statistical test — it's a sanity check that makes the system credible.

**Priority**: **MEDIUM**

---

### 4.3 MEDIUM: Regime Means Computed on Smoothed Probabilities

**File**: `src/regime_ml/regimes/labeling.py:40-41`
```python
Nk = np.maximum(gamma.sum(axis=0), 1e-12)
mu_k = (gamma.T @ X) / Nk[:, None]
```

Using smoothed (non-causal) probabilities to compute regime means includes future information. The "mean stress level in Regime 2" is computed using data points weighted by probabilities that depend on future observations. This is fine for interpretation but misleading if you intend to use these means for online classification.

**Fix**: Clearly document that labeling uses smoothed probabilities (interpretation only). For any live signal, use filter_proba-weighted means.

**Priority**: **MEDIUM**

---

## 5. Research Rigor & Reproducibility

### 5.1 HIGH: No Experiment Tracking

There is no logging of:
- Which model configuration was tested
- Which features were selected
- What the train/test split was
- What the random seed was
- What the log-likelihood at convergence was
- What the timestamp was

Every run is fire-and-forget via notebooks.

**Fix**: Implement a lightweight experiment tracker (even just a JSON/CSV log file). Each model fit should log: `{timestamp, model_id, n_regimes, covariance_type, n_features, feature_names, random_seed, split_date, converged, log_likelihood, n_iter_actual, scaler_params, all_evaluation_metrics}`.

**Priority**: **HIGH**

---

### 5.2 HIGH: Test Suite Is Essentially Empty

**Files**: `tests/conftest.py` (0 lines), `tests/test_transforms.py` (0 lines), `tests/test_registry.py` (0 lines), `tests/test_transform_parser.py` (21 lines, 3 tests)

The entire test suite is 3 tests covering only the transform parser. Zero tests for:
- Data pipeline stages
- Staleness tracking
- Calendar alignment
- HMM fitting/prediction
- Evaluation metrics
- Model selection
- Labeling

**Fix**: At minimum, add tests for:
1. Staleness tracking produces correct `is_new_data` flags
2. Forward-fill respects calendar boundaries
3. HMM `filter_proba` sums to 1 at each timestep
4. HMM `filter_proba` is causal (perturbing future data doesn't change past probabilities)
5. Evaluation metrics are bounded correctly
6. Model selection hard filters reject degenerate models
7. Feature names parse correctly from YAML

**Priority**: **HIGH**

---

### 5.3 MEDIUM: `random_state=42` Hardcoded in Multiple Places

**Files**: `src/regime_ml/regimes/hmm.py:174`, `src/regime_ml/regimes/hmm.py:15`

Both `HMMRegimeDetector` and `initialise_emissions` default to `random_state=42`. This is fine for reproducibility but the default should be `None` (random) with 42 used only when explicitly testing. The current setup gives a false sense of robustness — the researcher always gets the same result but doesn't know if it's stable.

**Priority**: **MEDIUM**

---

### 5.4 MEDIUM: No Pipeline Determinism Guarantee

The data pipeline uses `tqdm` and `print` for progress, but there's no hash/checksum of intermediate outputs. If the FRED data is updated upstream, rerunning the pipeline produces different results with no record of what changed.

**Fix**: Hash each pipeline stage output and log it. This enables detecting when upstream data changes.

**Priority**: **MEDIUM**

---

## 6. Structural & Code Quality Issues

### 6.1 MEDIUM: `get_top_features` Returns Hardcoded Feature Names

**File**: `src/regime_ml/features/macro/selection.py:19-71`

This function is a 70-line hardcoded list masquerading as a function. If any transform parameter changes in `regime_universe.yaml`, the feature names here silently become stale and won't match actual features.

**Fix**: Derive the feature ranking programmatically (e.g., by ANOVA R² from a pilot HMM run), or at minimum validate that all returned feature names exist in the actual feature set.

**Priority**: **MEDIUM**

---

### 6.2 MEDIUM: `build_featuregroup_map` Loads Data from Disk on Every Call

**File**: `src/regime_ml/data/macro/build_featuregroup_map.py:5-6`
```python
macro_cfg = load_configs()["macro_data"]["regime_universe"]
df_group = load_dataframe(macro_cfg["raw_path"])
```

This function is called from `evaluation.py` and `labeling.py`, potentially multiple times per model comparison. Each call reads YAML configs and loads a parquet file from disk.

**Fix**: Cache the result or pass the mapping as a parameter.

**Priority**: **MEDIUM**

---

### 6.3 MEDIUM: `ChainedTransform._compute` Bypasses Staleness

**File**: `src/regime_ml/features/common/transforms/base.py:114-118`
```python
def _compute(self, series: pd.Series) -> pd.Series:
    result = series
    for transform in self.transforms:
        result = transform._compute(result)
    return result
```

`ChainedTransform._compute` calls `_compute` on each child (not `transform`), bypassing staleness handling. However, `ChainedTransform.transform` correctly calls each child's `transform` method. The risk is that if anyone calls `chain._compute(series)` directly, staleness is silently ignored.

**Fix**: Override `_compute` to raise `NotImplementedError("Use transform() for ChainedTransform")` or make it call `transform()` on children.

**Priority**: **MEDIUM**

---

### 6.4 LOW: `test_get_feature_names` Is Wrong

**File**: `tests/test_transform_parser.py:16-21`
```python
def test_get_feature_names():
    parser = TransformParser()
    chain = parser.parse_chain([{"diff": {"periods": 21}}, {"z_score": {"window": 252}}])
    feature_names = parser.get_feature_names("vix", [chain])
    assert len(feature_names) == 2
    assert feature_names[0] == "diff_21"
    assert feature_names[1] == "z_score_252"
```

`get_feature_names` returns one name per chain, not per transform. A chain of [Diff, ZScore] should produce 1 feature name (e.g., `vix_diff_21_zscore_252`), not 2. The assertion `len(feature_names) == 2` is wrong — this is testing that a single chain produces 2 names, which contradicts the implementation at line 134 of `transform_parser.py` where each *chain* produces one name.

**Fix**: Fix the test to `assert len(feature_names) == 1` and `assert feature_names[0] == "vix_diff_21_zscore_252"`.

**Priority**: **LOW**

---

### 6.5 LOW: Pickle Serialization for Model Persistence

**File**: `src/regime_ml/regimes/hmm.py:458-459`
```python
with open(path, 'wb') as f:
    pickle.dump(self, f)
```

Pickle is fragile — if you rename a class, change its module path, or upgrade a dependency, old models become unloadable. It's also a security risk (arbitrary code execution on load).

**Fix**: Serialize model parameters (transition matrix, means, covariances, scaler parameters) as numpy arrays / JSON. Reconstruct the model from parameters on load.

**Priority**: **LOW**

---

### 6.6 LOW: Magic Numbers Throughout

- `alignment.py:86`: `days_since` calculation has no boundary handling
- `evaluation.py:81`: `n_mix=20` default for mixing diagnostic
- `selection.py:124`: `slack=0.75` in range scoring
- `selection.py:131`: persistence range `[20, 200]`
- `labeling.py:83-87`: all label coefficients (1.3, 1.1, -1.4, etc.)

Each of these should either be configurable via YAML or documented with economic justification.

**Priority**: **LOW**

---

## 7. Pre-Deployment Quant Hardening Checklist

### Before Conditioning Equity ML Models on Regimes:
1. **Fix lookahead bias in macro data** — use vintage data or apply publication lags (Section 1.2)
2. **Fix staleness detection** — use actual observation flags, not value-change detection (Section 1.1)
3. **Ensure scaler is IS-only** — verify no OOS leakage (Section 1.3)
4. **Add multi-seed initialization** — current results may be a local optimum (Section 2.3)
5. **Add BIC/AIC** — without complexity penalty, model selection is biased toward overfitting (Section 3.1)
6. **Increase OOS weight** — 5% is negligible (Section 3.2)
7. **Use filter_proba for OOS evaluation** — smooth_proba leaks within-window information (Section 1.4)
8. **Validate Gaussian assumption** — add QQ diagnostics, consider t-emissions (Section 2.1)
9. **Check convergence** — `model.monitor_.converged` (Section 2.2)

### Before Publishing to GitHub:
1. **Remove hardcoded paths** — `/Users/alecmitchell-thomson/...` appears in loaders.py and YAML (Section 1.8)
2. **Fix empty test files** — conftest.py, test_transforms.py, test_registry.py are all 0 bytes (Section 5.2)
3. **Fix the broken test** — `test_get_feature_names` asserts wrong values (Section 6.4)
4. **Fix typo** — "Policy-Contstrained" (Section 4.1)
5. **Remove `.env` from repo** — it's listed in the project and likely contains FRED API keys
6. **Add CI/CD** — no GitHub Actions for tests/lint

### Before Showing in an Interview:
1. A senior quant would immediately ask: **"How do you handle FRED revisions?"** — you have no answer right now
2. A senior quant would ask: **"Did you test multiple initializations?"** — single seed is a red flag
3. A senior quant would ask: **"Where is your BIC?"** — fundamental model selection tool, completely absent
4. A senior quant would ask: **"What's your OOS protocol?"** — a single split with 5% weight won't impress
5. The empty test suite signals this hasn't been validated beyond notebooks
6. The hardcoded feature ranking looks like the output was reverse-engineered to match expectations

### What Makes This Look Like a Student Project:
- Empty test files committed to git
- `print()` statements instead of `logging`
- Hardcoded personal paths in production code
- No experiment tracking — results live only in notebook memory
- Labels with trailing whitespace and typos
- Magic numbers without justification
- Pickle serialization
- `# type: ignore` comments scattered throughout (17+ occurrences)

### What Would Get This Rejected:
- **No revision handling** — any quant PM with FRED experience will flag this immediately
- **Single-seed HMM** — standard practice is multi-start
- **No BIC** — the most basic model selection criterion for mixture models
- **Gaussian assumption without validation** — especially with VIX in the feature set
- **5% OOS weight** — signals the developer doesn't actually trust their OOS evaluation

---

**Summary**: The architecture is well-structured and the staleness-aware transform framework is a genuinely good idea that most ML pipelines miss. The HMM implementation (especially the hand-rolled causal filter) shows technical competence. However, the system has several critical methodological gaps (revision handling, multi-seed, BIC, OOS protocol) that would be caught in any professional review. Fix these before building downstream models — regime conditioning is only as reliable as the regime labels, and right now those labels are not defensible.

---

## 8. Implementation Order

Ordered to unblock downstream work as fast as possible. Each phase can be started once the previous is done. Within a phase, items marked **[QUICK]** are low-effort and should be done first.

---

### Phase A — Data Integrity (fix before touching any model)

These create lookahead bias or corrupt the feature values that everything downstream depends on.

| # | Issue | Section | Effort |
|---|-------|---------|--------|
| A1 | Fix staleness detection — use pre-fill row flags, not value-change comparison | 1.1 | Medium |
| A2 | Add `release_lag_days` per series in `regime_universe.yaml`; shift series dates forward before use | 1.2 | High |
| A3 | Enforce IS-only scaler fitting; assert scaler never sees OOS dates | 1.3 | Medium |
| A4 | Replace `smooth_proba` with `filter_proba` for all OOS evaluation | 1.4 | **[QUICK]** |
| A5 | Add `max_staleness_days` per frequency; mark NaN instead of indefinite ffill | 1.5 | Medium |
| A6 | Vectorize `days_since_update` using ffill of last-update date index | 1.6 | **[QUICK]** |

---

### Phase B — HMM Model Fixes

These affect whether the model produces valid, stable, reproducible regime sequences.

| # | Issue | Section | Effort |
|---|-------|---------|--------|
| B1 | Check `model.monitor_.converged` after fit; warn and log LL trace | 2.2 | **[QUICK]** |
| B2 | Add Cholesky jitter (`C + 1e-8 * I`) in `filter_proba` | 2.5 | **[QUICK]** |
| B3 | Run N initializations (10–20 seeds); keep highest-LL model passing degeneracy filters | 2.3 | Medium |
| B4 | Replace ad-hoc `1e-6` eigenvalue floor with Ledoit-Wolf shrinkage for cluster covariances | 2.4 | Medium |
| B5 | Add QQ-plots per regime; evaluate t-emission alternative for VIX/NFCI | 2.1 | Medium |
| B6 | Implement Hungarian-algorithm label alignment for cross-model/cross-seed comparison | 2.6 | Medium |
| B7 | Replace `n_iter=1000` with `n_iter=500, tol=1e-4`; rely on convergence check from B1 | 2.7 | **[QUICK]** |

---

### Phase C — Model Selection

These affect whether the best model is actually selected.

| # | Issue | Section | Effort |
|---|-------|---------|--------|
| C1 | Increase OOS weight from 5% to ≥25% in `final_score` formula | 3.2 | **[QUICK]** |
| C2 | Compute BIC = -2·LL + k·log(T) for all candidates; add as hard filter or scoring component | 3.1 | Medium |
| C3 | Add documented economic justification for hard-filter thresholds (min_share, max_implied_duration, etc.) | 3.4 | **[QUICK]** |
| C4 | Replace rank-based scoring with z-score or min-max normalization for continuous metrics | 3.5 | Medium |
| C5 | Add `n_features` as a grid dimension in `compare_hmm_models`; cross-validate feature count selection | 3.3 | High |
| C6 | Implement expanding-window or rolling-window cross-validation; score by average OOS across folds | 3.6 | High |

---

### Phase D — Labels & Regime Interpretation

These affect whether the output regimes are credible and correctly labelled.

| # | Issue | Section | Effort |
|---|-------|---------|--------|
| D1 | Fix typo ("Contstrained") and trailing whitespace in label strings | 4.1 | **[QUICK]** |
| D2 | Make label set dynamic based on `n_regimes`; handle K≠4 cases | 4.1 | Medium |
| D3 | Document that `labeling.py` uses smoothed probabilities (interpretation only, not for live signals) | 4.3 | **[QUICK]** |
| D4 | Add `validate_against_episodes()` — check 2008 GFC, 2020 COVID, 2022 hike cycle, 2003–2007 expansion | 4.2 | Medium |

---

### Phase E — Tests & Reproducibility

These make the system verifiable and prevent regressions.

| # | Issue | Section | Effort |
|---|-------|---------|--------|
| E1 | Fix broken test: `test_get_feature_names` should assert `len==1` and correct name format | 6.4 | **[QUICK]** |
| E2 | Implement lightweight experiment logger: JSON record per fit with timestamp, config, seed, LL, metrics | 5.1 | Medium |
| E3 | Add core test suite: staleness flags, ffill boundaries, `filter_proba` causality, metric bounds, hard-filter rejection | 5.2 | High |
| E4 | Change `random_state` defaults to `None`; require explicit seed when testing | 5.3 | **[QUICK]** |
| E5 | Hash pipeline stage outputs; log hash with each run to detect upstream FRED data changes | 5.4 | Medium |

---

### Phase F — Code Quality & Portability

These block publishing or collaborating on the repo.

| # | Issue | Section | Effort |
|---|-------|---------|--------|
| F1 | Replace hardcoded `/Users/alecmitchell-thomson/...` paths with env-var resolution; raise clear error if unset | 1.8 | **[QUICK]** |
| F2 | Validate that all names returned by `get_top_features()` exist in the actual feature set at runtime | 6.1 | **[QUICK]** |
| F3 | Cache `build_featuregroup_map()` result (lru_cache or pass mapping as parameter) | 6.2 | **[QUICK]** |
| F4 | Override `ChainedTransform._compute` to raise `NotImplementedError`; enforce use of `transform()` | 6.3 | **[QUICK]** |
| F5 | Add post-selection correlation check in feature pipeline; drop or PCA-combine features above 0.85 threshold | 1.7 | Medium |
| F6 | Replace pickle serialization with parameter-level JSON + numpy for model persistence | 6.5 | Medium |
| F7 | Document or parameterize magic numbers (`n_mix=20`, `slack=0.75`, persistence range `[20,200]`, label coefficients) | 6.6 | Medium |

---

### Recommended Sprint Order

For immediate interview/deployment readiness, attack in this sequence:

1. **Day 1 — All QUICK items**: A4, A6, B1, B2, B7, C1, C3, D1, D3, E1, E4, F1, F2, F3, F4 — these are low-effort, high-signal fixes that eliminate the most embarrassing issues.
2. **Week 1 — Phase A remainder**: A1, A3, A5, then A2 (ALFRED/lag handling is the hardest and most important).
3. **Week 2 — Phase B**: B3 (multi-seed) + B4 + B5 (QQ diagnostics).
4. **Week 2 — Phase C**: C1 already done; C2 (BIC), C4 (scoring), C3 documentation.
5. **Week 3 — Phase D + E**: Labels (D2, D4), then tests (E2, E3).
6. **Week 4 — Phase F + C5/C6**: Portability cleanup, then expanding-window CV if time allows.
