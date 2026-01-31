# Regime-Conditioned Machine Learning Trading System

A research-grade equity trading engine than detects market regimes and conditions predictive machine learning models on regime-specific dynamics to handle financial non-stationarity

## Motivation

Financial markets are non-stationary,: relationships between features and returns change across volatility, liquidity, and macro regimes. Global predictive models often fail when regime structure is ignored

This project explores a regime-aware approach to equity prediction, combining unsupervised regime detection with conditional machine learning models to improve robustness and risk-adjusted performance

## System Overview

The system is composed of four core layers:
1. Data ingestion and feature engineering
2. Market regime detection
3. Regime-conditioned predictive modelling
4. Portfolio construction and evaluation

## Project Roadmap

### Phase 1 - Data Pipeline

**Status:** Complete - Data processing and feature engineering implemented

A production-ready end-to-end pipeline that transforms raw macro data into model-ready features for regime detection.

---

#### **Stage 1: Data Processing** (`src/regime_ml/data/`)

**Data Sources:**
- **9 Macro Indicators**: VIX, DGS2, DGS10, T10Y3M, NFCI, PCEPILFE, CFNAI, INDPRO, ICSA
- **Mixed Frequencies**: Daily (4), Weekly (2), Monthly (3)
- **Period**: 2000-2026 (~6,500 business days)

**Processing Steps:**
1. **Load**: Ingest raw parquet files from FRED API cache
2. **Select & Clean**: Filter to configured series, handle duplicates, weekend releases
3. **Align**: Forward-fill to US business day calendar (excluding federal holidays)
4. **Track Staleness**: Flag actual observations vs forward-filled values (`is_new_data`)
5. **Validate**: Quality checks (types, ranges, gaps, outliers, temporal consistency)
6. **Save**: Export to `data/processed/macro_processed.parquet`

**Key Innovation - Staleness Tracking:**
```
Monthly data (CFNAI):
  Actual:  [100.0, NaN, NaN, ..., 102.5, NaN, ...]  # ~12 points/year
  Filled:  [100.0, 100.0, 100.0, ..., 102.5, 102.5, ...]  # Daily
  Staleness: [True, False, False, ..., True, False, ...]  # Track actual vs filled
```
This prevents computing statistics on forward-filled (stale) data—a critical issue most pipelines miss.

---

#### **Stage 2: Feature Engineering** (`src/regime_ml/features/`)

**Transform System:**
- **Composable Transforms**: Level, Diff, PctChange, YoY, MA, EMA, RollingStd, ZScore
- **YAML-Driven**: Declarative feature definitions in `configs/data/regime_universe.yaml`
- **Staleness-Aware**: Computes only on actual observations, then forward-fills results

**Feature Generation:**
```yaml
# Example: VIX features
vix:
  frequency: daily
  transforms:
    - [level, {z_score: {window: 63}}]           # Absolute stress level
    - [{diff: {periods: 5}}, {z_score: {window: 126}}]   # Short-term momentum
    - [{ma: {window: 21}}, {diff: {periods: 21}}, {z_score: {window: 252}}]  # Trend
```

**Pipeline Flow:**
1. **Parse**: Convert YAML → ChainedTransform objects
2. **Apply**: For each indicator:
   - Extract series + staleness flags
   - Apply transform chains (compute on actual data only)
   - Forward-fill to daily frequency
3. **Generate**: Create descriptive feature names (`VIXCLS_diff_5_zscore_126`)
4. **Validate**: 11-point quality check (index, nulls, distributions, correlations, stationarity)
5. **Save**: Export raw + model-ready features

**Validation Checks:**
1. Index properties (business days, sorted)
2. Missing values (burn-in analysis)
3. Z-score distributions (mean≈0, std≈1)
4. Outliers (>5σ events)
5. Monthly indicators (step functions)
6. Weekly indicators (~52 changes/year)
7. Daily indicators (high variation)
8. Feature variance (low-variance detection)
9. Temporal consistency (discontinuities)
10. Feature correlations (<0.70 threshold)
11. Stationarity (ADF test p<0.05)

**Output:**
- **Raw Features**: `data/features/macro_features_raw.parquet` (with burn-in nulls)
- **Model-Ready**: `data/features/macro_features_ready.parquet` (2005-2026, 25 features, no nulls)
- **Metadata**: `data/features/feature_metadata.yaml` (stats, frequency, outliers per feature)

**Feature Selection:**

For regime detection, the pipeline selects the **top n (currently 5) most important features** from the validated set based on regime discriminative power:
1. **T10Y3M_level** - Yield curve slope (recession signal)
2. **VIXCLS_level** - Equity volatility (risk-on/risk-off)
3. **NFCI_level** - Financial conditions (credit stress)
4. **PCEPILFE_yoy_12** - Inflation (policy regime)
5. **CFNAI_level** - Economic activity (expansion/contraction)

This core set balances **interpretability** (can name regimes), **statistical efficiency** (avoids curse of dimensionality in HMMs), and **coverage** (captures all regime dimensions: growth, inflation, volatility, credit, rates).

**Final Dataset:**
- **Features**: 5 core regime indicators (selected from 25 validated features)
- **Period**: 2005-01-03 to 2026-01-15 (~5,250 business days)
- **Quality**: Low-correlation (<0.70), stationary (ADF p<0.05), validated
- **Ready for**: HMM-based regime detection and regime-conditioned modeling

### Phase 2 - Regime Detection

**Status:** In Progress — HMM-based regime detection with evaluation, selection, and labeling implemented

Unsupervised regime inference using Hidden Markov Models with Gaussian emissions. The implementation follows a two-stage evaluation philosophy: macro-quality filters (Stage A) first, then downstream equity usefulness (Stage B).

---

#### **HMM Regime Detector** (`src/regime_ml/regimes/hmm.py`)

**Model:**
- Gaussian HMM with configurable covariance type (`full`, `diag`)
- KMeans-based initialization for emission means and covariances via `initialise_emissions()`
- Custom transition matrix initialization via `initialise_transitions(p_stay)`
- Train/test split with scaling (StandardScaler) fitted on training data only

**Probabilities:**
- **Filtered** `filter_proba(X)` — Causal, uses only past data. For live/backtest use.
- **Smoothed** `smooth_proba(X)` — Non-causal, uses full history. For diagnostics and labeling.
- **Forecast** `forecast_n_steps(proba, n)` — n-step-ahead regime forecast from transition matrix

**Key Methods:**
- `fit(X)`, `predict(X)` — numpy arrays (n_samples, n_features)
- `get_transition_matrix()`, `get_regime_means()`, `get_regime_covariances()`
- `save(path)`, `load(path)` — pickle serialization

---

#### **Stage A: Macro-Regime Quality** (`src/regime_ml/regimes/evaluation.py`)

**Metrics computed per model:**

| Metric | Function | Purpose |
|--------|----------|---------|
| Regime persistence | `evaluate_regime_stability()` | Mean duration, n_transitions, regime share balance |
| Entropy balance | `evaluate_entropy_balance()` | Penalise collapsed/unused regimes |
| Transition sanity | `evaluate_transmat_sanity()` | Implied duration, diagonal dominance, mixing |
| Macro coherence | `evaluate_macro_coherence()` | Mahalanobis distance between regime means, ANOVA R² |

**Model Comparison:** `compare_hmm_models(features, models)` evaluates multiple trained HMMs on full sample and separately on in-sample vs out-of-sample slices. Requires models dict with `model`, `n_features`, `scaler`, `split_date`.

---

#### **Model Selection** (`src/regime_ml/regimes/selection.py`)

`select_best_hmm_model(results, ...)` applies:

**Hard filters (reject):** invalid transition matrix, dead regimes (min_share), collapsed regimes (max_share), absorbing regimes (max_implied_duration), redundant regimes (maha_min below quantile), OOS robustness failures.

**Composite score:** weighted combination of macro_score (40%), transition_score (30%), stability_score (25%), oos_macro_score (5%).

**Output:** `best_model_id`, leaderboard DataFrame (top_n survivors), rejected DataFrame (model_id + reason).

---

#### **Regime Labeling** (`src/regime_ml/regimes/labeling.py`)

`label_regimes(X, proba, feature_names)` assigns interpretable names to regimes using macro group signatures (growth, inflation, rates, liquidity, stress). Returns `state_labels`, `state_group_scores`, `state_feature_means` for use in visualisation and reporting.

---

#### **Stage B: Equity Usefulness** (`src/regime_ml/regimes/evaluation.py`)

`equity_metrics_by_regime(px, regimes, ...)` computes per-regime equity metrics: ann_return, ann_vol, Sharpe, max_drawdown, up_day_frac. Uses **filtered** regime labels (causal). Stage B uplift testing (regime-conditioned vs unconditional models) is planned.

---

#### **Visualisation** (`src/regime_ml/regimes/visualisation.py`)

- `plot_regime_timeseries()` — Regime labels, feature values, regime probabilities over time
- `plot_regime_distributions()` — Feature box plots by regime
- `plot_transition_matrix()` — Heatmap of regime transition probabilities
- `plot_regime_periods()` — Gantt-style regime timeline
- `plot_regime_confidence()` — Classification confidence over time
- `create_regime_summary_table()` — Metrics summary
- `plot_ticker_by_regime()` — Equity performance by regime

---

#### **Configuration & Plan**

- **Evaluation plan:** `src/regime_ml/regimes/evaluation_plan.md` — Full two-stage evaluation philosophy, metrics, scoring, and selection workflow
- **Feature groups:** `build_featuregroup_map()` from `regime_ml.data.macro` maps features to macro categories for coherence metrics and labeling

### Phase 3 - Regime-Conditioned Models

- Separate predictive models per regime
- Tree-based baselines, followed by neural models
- Probabilistic outputs and uncertainty estimates

### Phase 4 - Portfolio and Evaluation

- Risk-aware signal aggregation
- Transaction cost modeling
- Regime-aware performance attribution

## Project Structure

```
regime-conditioned-equity-ml/
│
├── configs/
│   └── README.md
│
├── data/
│   └── raw/
│   └── features/
│
├── src/
│   └── data/
│   └── features/
│   └── regimes/
│   └── models/
│   └── utils/
│
├── notebooks/
├── scripts/
├── tests/
├── pyproject.toml
└── README.md
```