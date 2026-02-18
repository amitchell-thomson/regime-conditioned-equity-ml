# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Regime-conditioned equity ML trading system. Detects market regimes via Hidden Markov Models and conditions predictive ML models on regime-specific dynamics to handle financial non-stationarity. Phases 1-2 (data pipeline, regime detection) are complete; Phases 3-4 (regime-conditioned modeling, portfolio evaluation) are in progress.

## Commands

```bash
# Install dependencies (uv is the package manager)
uv sync

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_transforms.py -v

# Run tests matching a pattern
pytest -k "test_transform" -v

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
ruff check --fix src/ tests/
```

## Architecture

### Package Layout

Source code lives in `src/regime_ml/` (src layout). The system is a staged pipeline:

**Raw Data → Load → Select → Clean → Align → Features → Regime Detection → (future) Models → Portfolio**

Key modules:
- `data/macro/` — Data pipeline: load FRED parquet files, select configured series, clean, align to business day calendar with staleness tracking, validate
- `features/common/transforms/` — Composable transform framework (BaseTransform, TransformRegistry factory, ChainedTransform). Transforms are declared in YAML and chained
- `features/macro/` — Apply transform chains to macro data, validate features (11-point quality check), select top-N features
- `regimes/` — HMM regime detector (`hmm.py`), evaluation metrics (`evaluation.py`), two-stage model selection with hard filters + soft ranking (`selection.py`), interpretable labeling (`labeling.py`), visualization (`visualisation.py`)
- `utils/config.py` — Centralized YAML config loader (`load_configs()`)

### Configuration

All major settings are YAML-based in `configs/`:
- `data/regime_universe.yaml` — 9 macro indicators with transform chain definitions and frequencies
- `regimes/regime_config.yaml` — HMM parameters, initialization, evaluation thresholds
- `models/model_config.yaml` — ML model hyperparameters (future)

### Staleness-Aware Processing

Critical design pattern: low-frequency data (monthly, weekly) is forward-filled to daily business days, but an `is_new_data` flag tracks actual observations vs. fills. Transforms with `staleness_mode='strict'` (default) compute only on actual data points, then forward-fill results. This prevents computing statistics on stale data.

### Transform Chain Pattern

Features are built by chaining transforms declared in YAML:
```yaml
vix:
  transforms:
    - [level, {z_score: {window: 63}}]
    - [{diff: {periods: 5}}, {z_score: {window: 126}}]
```

`TransformRegistry` (factory pattern) instantiates transforms by name. `TransformParser` converts YAML configs into `ChainedTransform` objects.

### HMM Regime Detection

`HMMRegimeDetector` wraps hmmlearn with KMeans-based initialization. Two probability modes:
- `smooth_proba()` — Non-causal (uses full history) for analysis
- `filter_proba()` — Causal (forward recursion only) for trading signals

### Model Selection

Two-stage: hard filters (valid transition matrix, min/max regime share, OOS robustness) followed by soft weighted ranking (macro score, transition score, stability score, OOS macro score).

## Data

Data lives in `data/` (gitignored). Raw macro data comes from FRED API. Final regime features are 5 selected indicators (T10Y3M, VIXCLS, NFCI, PCEPILFE, CFNAI) covering 2005-2026, stored as parquet.

Feature naming convention: `{INDICATOR}_{transform_chain}` (e.g., `VIXCLS_diff_5_zscore_126`).

## Tech Stack

- Python 3.13+, uv package manager
- pandas/numpy for data, hmmlearn for HMMs, scikit-learn for preprocessing
- pytest for tests, black for formatting, ruff for linting
- Jupyter notebooks in `notebooks/` for analysis (numbered sequentially)
