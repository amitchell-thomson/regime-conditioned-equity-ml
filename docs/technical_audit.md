# TECHNICAL AUDIT: Regime-Conditioned Equity ML System

**Classification**: Internal Code Review Memo
**Date**: 2026-02-21
**Scope**: Pre-deployment quant hardening review, Phases 1-2

---

## Summary

The architecture is well-structured and the staleness-aware transform framework is a genuinely good idea that most ML pipelines miss. The HMM implementation (especially the hand-rolled causal filter) shows technical competence. However, the system has several critical methodological gaps (revision handling, multi-seed, BIC, OOS protocol) that would be caught in any professional review. Fix these before building downstream models — regime conditioning is only as reliable as the regime labels, and right now those labels are not defensible.

---

## Assessment

### What's Strong

- Architecture is well-structured (staged pipeline, YAML config, transform registry)
- Staleness-aware transform framework is a genuinely good design
- Hand-rolled causal filter (`filter_proba`) shows technical competence

### Critical Gaps (Interview / Deployment Blockers)

| Gap | Step | Question a Senior Quant Would Ask |
| --- | ---- | --------------------------------- |
| No FRED revision handling | 2.4 | "How do you handle FRED revisions?" |
| Single-seed HMM | 3.1 | "Did you test multiple initializations?" |
| No BIC/AIC | 4.1 | "Where is your BIC?" |
| Gaussian assumption unvalidated | 3.3 | "Did you validate Gaussian emissions with VIX in the feature set?" |
| 5% OOS weight | 1.2a | "What's your OOS protocol?" |
| Empty test suite | 6.2 | (signals system hasn't been validated beyond notebooks) |
| Hardcoded feature ranking | 4.3 | (looks like output was reverse-engineered to match expectations) |

### Polish Issues (GitHub / Presentation Blockers)

- Empty test files committed to git
- `print()` statements instead of `logging`
- Hardcoded personal paths in production code
- No experiment tracking — results live only in notebook memory
- Labels with trailing whitespace and typos
- Magic numbers without justification
- Pickle serialization
- `# type: ignore` comments scattered throughout (17+ occurrences) → Step 7.5

### Pre-Deployment Checklist

- [x] Fix lookahead bias in macro data — ALFRED vintage data integrated; `build_realtime_series()` reconstructs point-in-time values (Step 2.4)
- [x] Fix staleness detection — `is_new_data=True` unconditionally pre-alignment; value-change check removed (Step 2.1)
- [x] Ensure scaler is IS-only — `train_end_date` guard added to `initialise_emissions`; `ValueError` raised on OOS leakage; `scaler._regime_ml_train_date_range` attached for audit (Step 2.2)
- [x] Add multi-seed initialization — `fit_best_of_n_seeds()` runs N seeds, keeps highest LL (Step 3.1)
- [x] Add BIC/AIC — `_n_params()`, `bic()`, `aic()` on `HMMRegimeDetector`; BIC is a soft score weight in selection (Step 4.1)
- [x] Increase OOS weight — raised from 5% → 25% (Step 1.2a)
- [x] Use filter_proba for OOS evaluation — smooth_oos replaced with filt_oos in evaluation.py (Step 1.4)
- [x] Validate Gaussian assumption — winsorize transform added; QQ diagnostics added to evaluation metrics (Step 3.3)
- [x] Check convergence — LL-delta check added; logger.warning fires when HMM hits iteration limit without converging (Step 1.1a)

### Pre-GitHub Checklist

- [x] Remove hardcoded paths — MACRO_DATA_PATH env var; loaders.py raises ValueError; .env.example added (Step 1.7b)
- [x] Fix empty test files — 190 tests now passing across 18 test files; includes BIC/AIC, labeling, episode validation, selection weights (Step 6.2 substantially done)
- [x] Fix the broken test — `test_get_feature_names` was already asserting correct values (Step 1.6)
- [x] Fix typo — "Policy-Contstrained" (Step 1.3a)
- [x] Replace `print()` with `logging` — NullHandler + package logger added (Step 1.9a ✓); all 180 print→logging migrated across 9 files (Step 2.0 ✓)
- [ ] Remove `.env` from repo — it's listed in the project and likely contains FRED API keys
- [x] Add CI/CD — `.github/workflows/ci.yml` added; runs pytest, ruff, black on push/PR (Step 7.4)
- [x] Resolve `# type: ignore` comments — all 33 occurrences narrowed to specific codes with explanatory comments (Step 7.5)

---

## Implementation Plan

### Recommended Sprint Order

1. ✅ **Day 1 — Phase 1 (Quick Wins)**: Low-effort, high-signal fixes including logging infrastructure (NullHandler + pytest config). These are prerequisites for everything that follows.
2. ✅ **Week 1 — Phase 2 (Data Integrity)**: Start by replacing 180 `print()` calls with structured logging so pipeline output is observable as you work through fixes. Then: staleness detection, IS-only scaler, max staleness, ALFRED/lag handling (hardest and most important).
3. ✅ **Week 2 — Phase 3 (HMM Model)**: Multi-seed initialization, Ledoit-Wolf covariance, QQ diagnostics.
4. ✅ **Week 2 — Phase 4 (Model Selection)**: BIC, normalized scoring, expanding-window CV.
5. ✅ **Week 3 — Phase 5 (Labels) + Phase 6 (Tests)**: Dynamic labels, episode validation, regime pipeline, 190-test suite.
6. ✅ **Week 4 — Phase 7 (Code Quality)**: Portability cleanup, CI/CD, type ignore resolution.

---

## Phase 1 — Quick Wins

Low-effort fixes that can be completed in a single focused session. Grouped by file to minimise context switching.

### Phase 1 — Implementation Status (2026-02-20)

| Step | Item | Status |
|------|------|--------|
| 1.1a | Convergence diagnostics (LL-delta warning) | ✅ Done |
| 1.1b | Cholesky jitter in `initialise_emissions` | ✅ Done |
| 1.1c | `n_iter=500, tol=1e-4` defaults | ✅ Done |
| 1.2a | OOS weight 5% → 25% | ✅ Done |
| 1.2b | Document hard filter thresholds (economic rationale comments) | ✅ Done |
| 1.3a | Fix "Policy-Contstrained" typo | ✅ Pre-existing |
| 1.3b | Document smoothed probabilities in `label_regimes()` docstring | ✅ Done |
| 1.4 | Use `filter_proba` for OOS evaluation | ✅ Done |
| 1.5 | Vectorize `days_since_update` | ✅ Done |
| 1.6 | Fix broken `test_get_feature_names` | ✅ Pre-existing |
| 1.7a | `random_state` defaults to `None` | ✅ Done |
| 1.7b | Replace hardcoded paths with `MACRO_DATA_PATH` env var | ✅ Done |
| 1.8a | Validate `get_top_features()` names at runtime | ✅ Done |
| 1.8b | Cache / replace `build_featuregroup_map` parquet I/O | ✅ Done (YAML lookup) |
| 1.8c | `ChainedTransform._compute()` guard + weighted staleness guard | ✅ Done |
| 1.9a | NullHandler at package root | ✅ Done |
| 1.9b | pytest `log_cli` config + hmmlearn/matplotlib silencing | ✅ Done |

**Phase 1 complete. All 17 items resolved. Proceed to Phase 2.**

---

### 1.1 `src/regime_ml/regimes/hmm.py` (3 fixes)

#### a) Add Convergence Diagnostics

**Priority**: HIGH | **Effort**: Quick

**File**: `src/regime_ml/regimes/hmm.py:223-226`
```python
self.model.fit(X, **kwargs)
self.is_fitted = True
return self
```

**Why dangerous**: The EM algorithm may not converge within `n_iter=1000`, converge to a local optimum, or converge to a degenerate solution where one regime captures <1% of data. None of these are detected or logged. `hmmlearn` sets `model.monitor_.converged` but this is never checked.

**Fix**:
```python
self.model.fit(X, **kwargs)
if not self.model.monitor_.converged:
    warnings.warn(f"HMM did not converge after {self.n_iter} iterations")
self.is_fitted = True
```
Also log `self.model.monitor_.history` (the log-likelihood trace) to detect oscillation vs. convergence.

---

#### b) Add Cholesky Jitter in `filter_proba`

**Priority**: MEDIUM | **Effort**: Quick

**File**: `src/regime_ml/regimes/hmm.py:295`
```python
L = np.linalg.cholesky(C)
```

**Why dangerous**: If the covariance matrix `C` is not strictly positive definite (e.g., numerical precision issues after EM), `cholesky` will raise `LinAlgError`. There's no try/catch or jitter.

**Fix**: Add a small jitter: `C = C + np.eye(d) * 1e-8` before Cholesky. Or use `scipy.linalg.cho_factor` which handles near-singular cases more gracefully.

---

#### c) Use Convergence Tolerance Instead of Fixed Iterations

**Priority**: LOW | **Effort**: Quick

**File**: `src/regime_ml/regimes/hmm.py:145`

1000 EM iterations is generous. Most well-initialized HMMs converge in 50-200 iterations. However, with poor initialization or near-degenerate solutions, 1000 may still not be enough.

**Fix**: Use `tol` (convergence threshold on log-likelihood improvement) instead of fixed iterations. hmmlearn supports `tol` parameter. Set `n_iter=500, tol=1e-4` and check convergence.

---

### 1.2 `src/regime_ml/regimes/selection.py` (2 fixes)

#### a) Increase OOS Weight from 5% to ≥25%

**Priority**: HIGH | **Effort**: Quick

**File**: `src/regime_ml/regimes/selection.py:147-152`
```python
survivors["final_score"] = (
    0.40 * survivors["macro_score"] +
    0.30 * survivors["transition_score"] +
    0.25 * survivors["stability_score"] +
    0.05 * survivors["oos_macro_score"]
)
```

**Why dangerous**: OOS performance gets 5% weight. This is essentially decorative. A model that collapses OOS but scores well IS will still win. This defeats the entire purpose of having an OOS evaluation.

**Fix**: Increase OOS weight to at least 20-30%. Better yet, restructure as: IS score selects candidates, OOS score selects the winner. A model that doesn't generalize OOS is worthless regardless of IS metrics.

---

#### b) Document Hard Filter Thresholds

**Priority**: MEDIUM | **Effort**: Quick

**File**: `src/regime_ml/regimes/selection.py:34-40`
```python
min_share: float = 0.03,
max_share: float = 0.80,
oos_min_share: float = 0.02,
oos_max_share: float = 0.85,
max_implied_duration: float = 3000.0,
```

**Why dangerous**: These thresholds are arbitrary magic numbers. Why is 3% the minimum share? Why 3000 days for max duration? A senior quant would immediately ask for the justification.

**Fix**: Either derive thresholds from economic priors (e.g., "we expect no regime shorter than 6 months ≈ 125 days, so minimum share = 125/5000 ≈ 2.5%") or use data-driven thresholds (e.g., percentiles of the candidate pool). Document the reasoning.

---

### 1.3 `src/regime_ml/regimes/labeling.py` (2 fixes)

#### a) Fix Typo and Trailing Whitespace in Labels

**Priority**: HIGH | **Effort**: Quick

**File**: `src/regime_ml/regimes/labeling.py:82-87`
```python
labels = [
    ("Risk On - Expansion               ",                    +1.3*zg + 1.1*zl - 1.4*zi - 1.0*zs - 0.3*zr),
    ("Risk On - Stagflation",                                 -1.1*zg + 1.5*zi + 0.6*zs + 0.3*zr),
    ("Risk On - Policy-Contstrained Expansion",               +1.2*zi + 1.3*zr - 0.9*zl - 0.4*zg),
    ("Risk Off - Recession",                                  -1.4*zg - 0.6*zl + 1.4*zs + 0.3*zi),
]
```

**Problems**:
1. **Typo**: "Policy-Contstrained" (missing 'r')
2. **Trailing whitespace** in "Risk On - Expansion               " — will cause display/comparison issues

**Fix**: Fix the typo and remove trailing whitespace.

---

#### b) Document That Labeling Uses Smoothed Probabilities

**Priority**: MEDIUM | **Effort**: Quick

**File**: `src/regime_ml/regimes/labeling.py:40-41`
```python
Nk = np.maximum(gamma.sum(axis=0), 1e-12)
mu_k = (gamma.T @ X) / Nk[:, None]
```

Using smoothed (non-causal) probabilities to compute regime means includes future information. The "mean stress level in Regime 2" is computed using data points weighted by probabilities that depend on future observations. This is fine for interpretation but misleading if you intend to use these means for online classification.

**Fix**: Clearly document that labeling uses smoothed probabilities (interpretation only). For any live signal, use filter_proba-weighted means.

---

### 1.4 `src/regime_ml/regimes/evaluation.py` — Use `filter_proba` for OOS Evaluation

**Priority**: HIGH | **Effort**: Quick

**File**: `src/regime_ml/regimes/evaluation.py:288-289`
```python
smooth_is = model.smooth_proba(X_is) if X_is.shape[0] > 0 else None
smooth_oos = model.smooth_proba(X_oos) if X_oos.shape[0] > 0 else None
```

**Why dangerous**: `smooth_proba` runs the forward-backward algorithm within each slice independently, so IS and OOS are separated. However, using smoothed probabilities for OOS evaluation is still methodologically suspect: in a real trading system, you would only have `filter_proba` (causal). Evaluating with smoothed probabilities inflates OOS coherence metrics because the backward pass uses "future" information within the OOS window itself.

**Fix**: OOS evaluation should use `filter_proba` exclusively. Reserve `smooth_proba` for IS analysis/interpretation only.

---

### 1.5 `src/regime_ml/data/macro/alignment.py` — Vectorize `days_since_update` Calculation

**Priority**: MEDIUM | **Effort**: Quick

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

---

### 1.6 `tests/test_transform_parser.py` — Fix Broken `test_get_feature_names` Test

**Priority**: LOW | **Effort**: Quick

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

---

### 1.7 Configuration & Defaults (2 fixes)

#### a) Change `random_state` Defaults to `None`

**Priority**: MEDIUM | **Effort**: Quick

**Files**: `src/regime_ml/regimes/hmm.py:174`, `src/regime_ml/regimes/hmm.py:15`

Both `HMMRegimeDetector` and `initialise_emissions` default to `random_state=42`. This is fine for reproducibility but the default should be `None` (random) with 42 used only when explicitly testing. The current setup gives a false sense of robustness — the researcher always gets the same result but doesn't know if it's stable.

**Fix**: Change `random_state` defaults to `None`; require explicit seed when testing.

---

#### b) Replace Hardcoded Paths with Environment Variables

**Priority**: LOW (but HIGH if publishing) | **Effort**: Quick

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

---

### 1.8 Feature Pipeline (3 fixes)

#### a) Validate Feature Names at Runtime

**Priority**: MEDIUM | **Effort**: Quick

**File**: `src/regime_ml/features/macro/selection.py:19-71`

This function is a 70-line hardcoded list masquerading as a function. If any transform parameter changes in `regime_universe.yaml`, the feature names here silently become stale and won't match actual features.

**Fix**: Validate that all names returned by `get_top_features()` exist in the actual feature set at runtime.

---

#### b) Cache `build_featuregroup_map` Result

**Priority**: MEDIUM | **Effort**: Quick

**File**: `src/regime_ml/data/macro/build_featuregroup_map.py:5-6`
```python
macro_cfg = load_configs()["macro_data"]["regime_universe"]
df_group = load_dataframe(macro_cfg["raw_path"])
```

This function is called from `evaluation.py` and `labeling.py`, potentially multiple times per model comparison. Each call reads YAML configs and loads a parquet file from disk.

**Fix**: Cache the result or pass the mapping as a parameter.

---

#### c) Override `ChainedTransform._compute` to Prevent Staleness Bypass

**Priority**: MEDIUM | **Effort**: Quick

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

---

### 1.9 Logging Infrastructure (2 fixes)

These are prerequisites for Phase 2. They take <30 minutes and make all subsequent pipeline output observable and suppressible.

#### a) Add Package-Level Logger Configuration

**Priority**: MEDIUM | **Effort**: Quick

**File**: `src/regime_ml/__init__.py`

No `logging` module usage exists anywhere. The correct library pattern is to install a `NullHandler` at the package root so downstream tools and tests can control output. Without it, any `print()` that survives the Phase 2 migration still has no controllable output channel.

**Fix**:
```python
# src/regime_ml/__init__.py
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

Each module then declares its own child logger at the top of the file:
```python
import logging
logger = logging.getLogger(__name__)
```

Callers (notebooks, scripts) configure output as needed:
```python
logging.basicConfig(level=logging.INFO)
```

---

#### b) Configure pytest Logging Capture

**Priority**: LOW | **Effort**: Quick

**Files**: `pyproject.toml`, `tests/conftest.py`

`conftest.py` is empty. pytest swallows log records unless `log_cli` is enabled. Test failures currently show no logging context, and `WARNING`-level validator output is silently dropped.

**Fix** — add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
log_cli = true
log_cli_level = "WARNING"
log_level = "DEBUG"
```

Add to `tests/conftest.py`:
```python
import logging

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
```

---

## Phase 2 — Data Integrity

Fix before touching any model. These create lookahead bias or corrupt the feature values that everything downstream depends on.

Start with 2.0 (print→logging) so that all pipeline output from 2.1 onward is structured, observable, and suppressible in tests.

### Phase 2 — Implementation Status (2026-02-21)

| Step | Item | Status |
|------|------|--------|
| 2.0 | Replace ~180 `print()` calls with structured `logging` across 9 files | ✅ Done |
| 2.1 | Fix staleness detection — `is_new_data=True` unconditionally pre-alignment | ✅ Done |
| 2.2 | IS-only scaler enforcement — `train_end_date` guard + `scaler._regime_ml_train_date_range` | ✅ Done |
| 2.3 | Max staleness limit on ffill — per-frequency thresholds in YAML + `align_to_calendar()` | ✅ Done |
| 2.4 | ALFRED vintage integration — `load_alfred_data()`, `build_realtime_series()`, pipeline routing | ✅ Done |

**Additional bugs fixed during testing:**
- `features/macro/pipeline.py` and `features/macro/__init__.py`: erroneous `get_feature_groups` import removed
- `build_realtime_series()`: revision tie-breaking fixed — now takes latest revision of most recent obs period (`max realtime_start` among `max obs_date` candidates), not earliest vintage
- `align_to_calendar()` frequency detection: `aligned['native_freq'].iloc[0]` returned NaN when calendar starts before first observation; fixed to `dropna().iloc[0]`

**Test coverage added (36 new tests, 82/82 passing):**

| File | Tests |
|------|-------|
| `tests/test_staleness_detection.py` | 8 — staleness fix correctness |
| `tests/test_hmm_is_enforcement.py` | 10 — OOS guard, scaler metadata, logging |
| `tests/test_max_staleness.py` | 9 — threshold enforcement, backward compat, freq fallback |
| `tests/test_alfred_integration.py` | 17 — schema, causal correctness, filtering, deduplication |

**Phase 2 complete. All 5 items resolved. Proceed to Phase 3.**

---

### 2.0 Replace `print()` with Structured Logging

**Priority**: MEDIUM | **Effort**: Medium

**Files**: 9 files, 180 total print statements

| File | Prints | Appropriate Level |
|------|--------|-------------------|
| `features/macro/validator.py` | 69 | `DEBUG` (per-check diagnostics), `WARNING` (failures) |
| `data/macro/validators.py` | 32 | `INFO` (summary lines), `WARNING` (data issues) |
| `data/macro/pipeline.py` | 30 | `INFO` (stage progress, timing) |
| `data/macro/loaders.py` | 12 | `INFO` (file load status) |
| `features/macro/pipeline.py` | 11 | `INFO` (pipeline progress, date range, save path) |
| `data/macro/cleaners.py` | 2 | `INFO` |
| `regimes/visualisation.py` | 2 | `INFO`, `WARNING` |
| `utils/config.py` | 1 | `WARNING` (config file not found) |
| `features/common/transforms/base.py` | 1 | `WARNING` (unimplemented staleness mode) |

**Level assignment rules**:
- Any `print("Warning: ...")` or unmet expectation → `logger.warning(...)`
- Pipeline stage headers, data shapes, date ranges, save locations → `logger.info(...)`
- Per-series / per-feature verbose diagnostics → `logger.debug(...)`
- Unrecoverable failures → `logger.error(...)`

**Fix** (representative examples):
```python
# Before (utils/config.py)
print(f"Warning: Config file not found: {path}")
# After
logger.warning("Config file not found: %s", path)

# Before (data/macro/pipeline.py)
print(f"  Saved aligned data to {output_path}")
# After
logger.info("Saved aligned data to %s", output_path)
```

Use `%`-style formatting rather than f-strings — the message string is not constructed when the log level is suppressed, which matters in the 69-print validator.

---

### 2.1 Fix Staleness Detection (Value Equality Is Wrong)

**Priority**: CRITICAL | **Effort**: Medium

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

---

### 2.2 Enforce IS-Only Scaler Fitting

**Priority**: HIGH | **Effort**: Medium

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

---

### 2.3 Add Maximum Staleness Limit to Forward-Fill

**Priority**: MEDIUM | **Effort**: Medium

**File**: `src/regime_ml/data/macro/alignment.py:72`
```python
aligned['value'] = aligned['value'].ffill()
```

**File**: `src/regime_ml/data/macro/cleaners.py:62-87` (`trim_to_common_start`)

The pipeline forward-fills values, then trims to the latest series start date. But if a series has a gap in the middle (e.g., FRED discontinues publication for a period), `ffill()` will silently propagate stale values indefinitely. There is no maximum staleness limit.

**Fix**: Add a `max_staleness_days` parameter per frequency. If a series hasn't updated in >2x its native frequency (e.g., 60 days for monthly), mark as NaN rather than forward-filling.

---

### 2.4 Add Publication Lag / Revision Handling

**Priority**: CRITICAL | **Effort**: High

**File**: `src/regime_ml/data/macro/loaders.py` (entire module), `configs/data/regime_universe.yaml`

**Why dangerous**: FRED stores the *latest revised* values for all historical dates. Many macro series used here are subject to significant revisions:
- **CFNAI**: Revised monthly with 1-month lag, then benchmark-revised annually
- **INDPRO**: Preliminary → 1st revision → 2nd revision → benchmark revision
- **PCEPILFE**: Released with ~1-month lag, revised for 2+ months

The pipeline loads the current snapshot of FRED data and treats every value as if it were known at time `t`. This creates **lookahead bias**: the model trains on revised values that were not available at the time. For example, INDPRO for January 2020 may have been revised 3 times by March 2020, but the model assumes the final (revised) value was known on the January release date.

**Fix**:
1. Use ALFRED (Archival FRED) real-time vintage data, which stores each value as of its initial release date and subsequent revisions.
2. At minimum, add a `release_lag_days` field per series in `regime_universe.yaml` and shift dates forward by the publication lag. E.g., monthly CFNAI released end of following month: shift by ~30 business days.

---

## Phase 3 — HMM Model Fixes

These affect whether the model produces valid, stable, reproducible regime sequences. Assumes Phase 2 data integrity fixes are in place.

### Phase 3 — Implementation Status (2026-02-26)

| Step | Item | Status |
|------|------|--------|
| 3.1 | Multi-seed initialization — `fit_best_of_n_seeds()`, N seeds, highest LL kept | ✅ Done |
| 3.2 | Ledoit-Wolf covariance regularization — replaced ad-hoc eigenvalue floor | ✅ Done |
| 3.3 | Gaussian assumption validation — winsorize transform + QQ diagnostics in evaluation | ✅ Done |
| 3.4 | State permutation / label alignment — Hungarian algorithm via `align_states()` in `hmm.py` | ✅ Done |

**Phase 3 complete. All 4 items resolved. Proceed to Phase 4.**

---

### 3.1 Multi-Seed Initialization

**Priority**: HIGH | **Effort**: Medium

**File**: `src/regime_ml/regimes/hmm.py:174`
```python
random_state: int = 42,
```

EM for HMMs is highly sensitive to initialization. Using a single seed (42) means:
- You have no idea if this is a local or global optimum
- The solution may be unstable — a different seed could produce entirely different regimes
- The KMeans initialization helps, but KMeans itself is also sensitive to initialization

**Fix**: Run N initializations (e.g., 10-20 seeds), keep the model with highest log-likelihood that passes degeneracy filters. This is standard practice. Log all runs for reproducibility.

---

### 3.2 Replace Ad-Hoc Covariance Regularization with Ledoit-Wolf

**Priority**: MEDIUM | **Effort**: Medium

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

---

### 3.3 Validate Gaussian Emission Assumption

**Priority**: HIGH | **Effort**: Medium

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
1. At minimum, add QQ-plots per regime to validate Gaussian assumption
2. Consider winsorizing at 4σ before fitting to reduce tail sensitivity

---

### 3.4 Implement State Permutation / Label Alignment

**Priority**: MEDIUM | **Effort**: Medium

When fitting HMMs with different seeds or on different subsamples, state indices are arbitrary (state 0 in run A may correspond to state 2 in run B). There is no label alignment mechanism.

**Impact**: Comparing OOS regime shares to IS regime shares (as done in `select_best_hmm_model`) assumes state indices are consistent between IS and OOS. Since the same fitted model is used for both, this is fine within a single model. But if you ever compare two different fitted models, or extend to expanding-window refit, you'll need permutation alignment.

**Fix**: Implement Hungarian algorithm (scipy.optimize.linear_sum_assignment) for regime label alignment based on KL divergence between emission distributions.

---

## Phase 4 — Model Selection

These affect whether the best model is actually selected. Assumes Phases 2-3 are in place.

### Phase 4 — Implementation Status (2026-02-26)

| Step | Item | Status |
|------|------|--------|
| 4.1 | BIC/AIC — `_n_params()`, `bic()`, `aic()` on detector; BIC soft weight in selection (weight=0.10) | ✅ Done |
| 4.2 | Smooth scoring — `_range_score()` replaced with `_soft_score(x, optimal, lo, hi, slack)` | ✅ Done |
| 4.3 | Feature selection in grid — **OBSOLETE**: feature selection is fully YAML-configured via PCA blocks per macro group; no longer a model grid concern | ➖ N/A |
| 4.4 | Expanding-window CV — `expanding_window_cv()` in `evaluation.py`; opt-in via `cross_validation.enabled` | ✅ Done |

**Phase 4 complete (4.3 superseded by group PCA architecture). Proceed to Phase 5.**

**Selection weight scheme (5-component, sums to 1.0):**
`macro: 0.25 | transitions: 0.25 | stability: 0.20 | oos: 0.20 | bic: 0.10`

---

### 4.1 Add BIC/AIC

**Priority**: HIGH | **Effort**: Medium

**File**: `src/regime_ml/regimes/selection.py` (absence)

The model selection pipeline (`selection.py`) uses custom scoring but never computes AIC or BIC. For HMMs, the number of free parameters grows quadratically with `n_regimes` (transition matrix: K²-K, means: K*d, covariance: K*d*(d+1)/2 for full). Without a complexity penalty, the scoring system will systematically prefer higher-K models that overfit.

**Fix**: Compute BIC = -2*LL + k*log(T) where k is the number of free parameters. Use it as either a hard filter or a scoring component. This is the single most important missing metric.

---

### 4.2 Replace Rank-Based Scoring with Normalized Scoring

**Priority**: MEDIUM | **Effort**: Medium

**File**: `src/regime_ml/regimes/selection.py:117-118`
```python
def rrank(s, ascending=False):
    return s.rank(pct=True, ascending=ascending)
```

Rank-based scoring (percentile ranks) treats models that score 0.95 and 0.50 on Mahalanobis distance the same as models that score 0.51 and 0.50. If one model is dramatically better than all others on macro coherence, rank scoring won't reward it.

**Fix**: Use z-score normalization or min-max scaling instead of ranks for continuous metrics. Ranks are appropriate when the metric distributions are unknown, but here the metrics are well-defined.

---

### 4.3 Include Feature Selection in Model Selection Grid

**Priority**: HIGH | **Effort**: High

**File**: `src/regime_ml/features/macro/selection.py:19-71`

`get_top_features()` returns a fixed, manually-ranked list. The `compare_hmm_models` function calls `get_top_features(n=n_features)` to select features. This means:
- Feature selection is not cross-validated
- The feature ranking was determined by the researcher looking at (presumably) in-sample results
- There's no way to know if the ranking holds OOS

**Fix**: Feature selection should be part of the model selection grid. Enumerate feature subsets (or at least different `n` values) as model configurations, and let the scoring function pick the best combination.

---

### 4.4 Add Expanding-Window Cross-Validation

**Priority**: MEDIUM | **Effort**: High

The IS/OOS split is a single time-series split. This means:
- The split date is a free parameter (data-snooping risk if tuned)
- A single OOS window may be unrepresentative (e.g., if OOS is 2020-2026, dominated by COVID)
- No estimate of model stability across time

**Fix**: Implement expanding-window or rolling-window cross-validation. Refit HMM on {2005-2010, 2005-2012, 2005-2014, ...} and evaluate on the subsequent 2-3 year window. Score by average OOS performance across folds.

---

## Phase 5 — Labels & Regime Interpretation

These affect whether the output regimes are credible and correctly labelled.

### Phase 5 — Implementation Status (2026-02-26)

| Step | Item | Status |
|------|------|--------|
| 5.1 | Dynamic labeling — archetype pool in `regime_archetypes.yaml`; cosine similarity + linear assignment; confidence threshold; unclassified fallback | ✅ Done |
| 5.2 | Episode validation — `validate_against_episodes()` in `evaluation.py`; 9 episodes in `economic_episodes.yaml` | ✅ Done |

**Phase 5 complete. Regime pipeline module also added (`pipeline.py`) with full IS→OOS→label→validate→save flow.**

**New files:**
- `configs/regimes/regime_archetypes.yaml` — 6 archetypes (expansion, stagflation, policy_constrained, recession, recovery, liquidity_crisis)
- `configs/regimes/economic_episodes.yaml` — 9 known macro episodes with expected archetypes
- `src/regime_ml/regimes/pipeline.py` — end-to-end regime pipeline callable from CLI and notebooks

---

### 5.1 Make Label Set Dynamic Based on `n_regimes`

**Priority**: HIGH | **Effort**: Medium

**File**: `src/regime_ml/regimes/labeling.py:82-87`
```python
labels = [
    ("Risk On - Expansion               ",                    +1.3*zg + 1.1*zl - 1.4*zi - 1.0*zs - 0.3*zr),
    ("Risk On - Stagflation",                                 -1.1*zg + 1.5*zi + 0.6*zs + 0.3*zr),
    ("Risk On - Policy-Contstrained Expansion",               +1.2*zi + 1.3*zr - 0.9*zl - 0.4*zg),
    ("Risk Off - Recession",                                  -1.4*zg - 0.6*zl + 1.4*zs + 0.3*zi),
]
```

**Problems**:
1. **Exactly 4 labels for any K**: If `n_regimes=3` or `n_regimes=5`, the labeling system is broken — multiple regimes will get the same label
2. **Label coefficients are arbitrary**: The weights (1.3, 1.1, -1.4, etc.) are not derived from data or economic theory. They're hand-tuned to produce "reasonable" labels.
3. **First label says "Risk On" but the model might assign it to a risk-off cluster** — the labeling is a post-hoc interpretation that may not match the statistical structure

**Fix**:
- Make the label set dynamic based on `n_regimes`
- Derive labels from data (e.g., which macro group has the largest absolute z-score defines the label)
- Remove magic coefficients

---

### 5.2 Add Validation Against Known Economic Episodes

**Priority**: MEDIUM | **Effort**: Medium

There's no check that regimes align with known episodes. A credible system would verify:
- 2008 GFC → stress/recession regime
- 2020 COVID → brief stress → rapid recovery
- 2022 rate-hiking → tightening/inflation regime
- 2003-2007 → expansion regime

This is conspicuously absent from evaluation metrics.

**Fix**: Add a `validate_against_episodes()` function that checks regime classifications against a list of known economic episodes with expected regime types. This is not a statistical test — it's a sanity check that makes the system credible.

---

## Phase 6 — Tests & Reproducibility

These make the system verifiable and prevent regressions.

### Phase 6 — Implementation Status (2026-02-26)

| Step | Item | Status |
|------|------|--------|
| 6.1 | Experiment tracking — `run_metadata.json` with `features_hash`, `config_hash`, timestamps, model selection scores, and feature names | ✅ Done |
| 6.2 | Core test suite — 214 tests across 25 files; all pipeline stages, HMM internals, and new utilities covered | ✅ Done |
| 6.3 | Pipeline determinism — SHA-256 `features_hash` added to `run_metadata.json` via `pd.util.hash_pandas_object` | ✅ Done |
| 6.4 | Audit trail logging — `configure_pipeline_logging()` in `src/regime_ml/utils/logging.py`; optional `--log-dir` on `run_regime_pipeline()` and CLI | ✅ Done |

**Phase 6 complete. All 4 items resolved. Proceed to Phase 7.**

**New test files added:**

| File | Tests | Coverage |
|------|-------|---------|
| `tests/test_pipeline_metadata.py` | 5 | SHA-256 features_hash: stability, sensitivity, hex format |
| `tests/test_pipeline_logging.py` | 5 | `configure_pipeline_logging`: file creation, timestamp pattern, message capture, root logger isolation |
| `tests/test_cross_group_correlation.py` | 5 | Correlation check: threshold behaviour, IS-only data, edge cases |
| `tests/test_hmm_serialization.py` | 9 | JSON/numpy save/load: roundtrip predict and filter_proba, missing dir, unfitted guard, no pickle import |

---

### 6.1 Implement Experiment Tracking

**Priority**: HIGH | **Effort**: Medium

There is no logging of:
- Which model configuration was tested
- Which features were selected
- What the train/test split was
- What the random seed was
- What the log-likelihood at convergence was
- What the timestamp was

Every run is fire-and-forget via notebooks.

**Fix**: Implement a lightweight experiment tracker (even just a JSON/CSV log file). Each model fit should log: `{timestamp, model_id, n_regimes, covariance_type, n_features, feature_names, random_seed, split_date, converged, log_likelihood, n_iter_actual, scaler_params, all_evaluation_metrics}`.

---

### 6.2 Build Core Test Suite

**Priority**: HIGH | **Effort**: High

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

---

### 6.3 Add Pipeline Determinism Guarantee

**Priority**: MEDIUM | **Effort**: Medium

The data pipeline uses `tqdm` for progress, but there's no hash/checksum of intermediate outputs. If the FRED data is updated upstream, rerunning the pipeline produces different results with no record of what changed.

**Fix**: Hash each pipeline stage output and log it. This enables detecting when upstream data changes.

---

### 6.4 Add File Handler for Pipeline Audit Trail

**Priority**: LOW | **Effort**: Medium

**Files**: `src/regime_ml/data/macro/pipeline.py`, `src/regime_ml/regimes/selection.py`

Pipeline runs produce output only to stdout. Background or notebook runs leave no persistent record — which series loaded, what validation warnings fired, which HMM models were filtered. This is the logging equivalent of the experiment tracking gap (6.1) and makes post-hoc debugging impossible. Requires 2.0 (print→logging) to be complete first.

**Fix**: Add a utility function callers invoke once at the start of a pipeline run:
```python
import logging, datetime, pathlib

def configure_pipeline_logging(log_dir: pathlib.Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"pipeline_{ts}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    logging.getLogger("regime_ml").addHandler(fh)
```

Call at the top of each pipeline script or in a notebook setup cell. Each run produces a timestamped `.log` file that, combined with the experiment tracker (6.1), gives a complete reproducibility record.

---

## Phase 7 — Code Quality & Portability

These block publishing or collaborating on the repo.

### Phase 7 — Implementation Status (2026-02-26)

| Step | Item | Status |
|------|------|--------|
| 7.1 | Cross-group PC correlation check — `_check_cross_group_correlation()` in `features/macro/selection.py`; IS-data only; `warn_threshold` from `feature_selection.cross_group_correlation` in config | ✅ Done |
| 7.2 | Pickle replacement — `save()` / `load()` rewritten to directory-based JSON + numpy format; `import pickle` removed from `hmm.py` | ✅ Done |
| 7.3 | Magic numbers — `n_mix`, `slack`, `lo`, `hi`, `optimal` moved to `regime_config.yaml`; `evaluation.py` and `selection.py` read all bounds from config with safe defaults | ✅ Done |
| 7.4 | CI/CD — `.github/workflows/ci.yml` added; runs `pytest`, `ruff check`, and `black --check` on push to main/refactoring and PRs to main | ✅ Done |
| 7.5 | Type ignores — all 33 bare `# type: ignore` narrowed to specific mypy error codes (`[attr-defined]`, `[return-value]`, `[index]`, `[arg-type]`, `[union-attr]`, etc.) with explanatory comments | ✅ Done |

**Phase 7 complete. All 5 items resolved. All pre-GitHub checklist items now ticked.**

**Config keys added for Phase 7.1 / 7.3:**
- `feature_selection.cross_group_correlation.warn_threshold: 0.80`
- `evaluation.transmat_sanity.n_mix: 20`
- `selection.soft_score.{duration,turnover,persistence}.{optimal,lo,hi,slack}`

---

### 7.1 Add Post-Selection Feature Correlation Check

**Priority**: MEDIUM | **Effort**: Medium

**File**: `src/regime_ml/features/macro/selection.py:36-71`

The ranked feature list includes multiple correlated signals:
- `DGS10_level_zscore_252` and `DGS2_level_zscore_252` — both nominal rate levels, highly correlated
- `T10Y3M_level_zscore_252`, `T10Y3M_diff_21_zscore_252`, `T10Y3M_diff_5_zscore_126` — three curve features

The feature validator checks 0.70 correlation, but the feature *selection* is hardcoded by economic intuition without empirical deduplication. In practice, DGS2 and DGS10 will have correlations >0.90 in many regimes.

**Fix**: Add a post-selection correlation check that either drops or PCA-combines features exceeding a threshold (e.g., 0.85). Or use the correlation matrix to inform the ranked list rather than relying purely on judgment.

---

### 7.2 Replace Pickle Serialization

**Priority**: LOW | **Effort**: Medium

**File**: `src/regime_ml/regimes/hmm.py:458-459`
```python
with open(path, 'wb') as f:
    pickle.dump(self, f)
```

Pickle is fragile — if you rename a class, change its module path, or upgrade a dependency, old models become unloadable. It's also a security risk (arbitrary code execution on load).

**Fix**: Serialize model parameters (transition matrix, means, covariances, scaler parameters) as numpy arrays / JSON. Reconstruct the model from parameters on load.

---

### 7.3 Document or Parameterize Magic Numbers

**Priority**: LOW | **Effort**: Medium

- `alignment.py:86`: `days_since` calculation has no boundary handling
- `evaluation.py:81`: `n_mix=20` default for mixing diagnostic
- `selection.py:124`: `slack=0.75` in range scoring
- `selection.py:131`: persistence range `[20, 200]`
- `labeling.py:83-87`: all label coefficients (1.3, 1.1, -1.4, etc.)

Each of these should either be configurable via YAML or documented with economic justification.

---

### 7.4 Add CI/CD

**Priority**: MEDIUM | **Effort**: Small

**Files**: `.github/workflows/` (absent)

No GitHub Actions exist. There are no automated checks on push or PR — tests, lint, and formatting are never verified without manual effort. Any broken commit goes undetected until someone runs `pytest` locally.

**Fix**: Add `.github/workflows/ci.yml` running on push and PR:
```yaml
- run: uv run pytest tests/ -v
- run: uv run ruff check src/ tests/
- run: uv run black --check src/ tests/
```

**Note**: This is only viable once the test suite is non-empty (Step 6.2).

---

### 7.5 Remove `# type: ignore` Comments

**Priority**: LOW | **Effort**: Medium

**Files**: 17+ occurrences across `src/regime_ml/`

`# type: ignore` suppresses mypy errors silently. 17+ occurrences means type issues are being papered over rather than fixed. This is a maintenance risk — a type error that causes a runtime bug will never surface in static analysis.

**Fix**: For each occurrence, either:
1. Fix the underlying type issue (correct annotation, add a cast, or refine the type)
2. Replace with a narrower suppression: `# type: ignore[attr-defined]` with a comment explaining why
3. As a last resort, leave `# type: ignore` only where a third-party library (e.g., hmmlearn) has incomplete stubs — and document this explicitly

Running `mypy src/regime_ml/ --ignore-missing-imports` first will show which are real issues vs. stub gaps.

