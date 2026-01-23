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

**Final Dataset:**
- **Features**: 25 low-correlation, stationary features
- **Categories**: Stress (7), Rates (8), Inflation (3), Growth (4), Labor (3)
- **Period**: 2005-01-03 to 2026-01-15 (~5,250 business days)
- **Ready for**: HMM-based regime detection and regime-conditioned modeling

### Phase 2 - Regime Detection

- Unsupervised regime inference (eg. HMMs, clustering)
- Volatility and correlation-based regime definitions
- Regime stability and transition dynamics

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