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

### Phase 1 - Data and Features

- Equity price and volume data (daily, could extend to intraday in the future)
- Return, volatility, trend and cross-sectional features
- Feature normalization and leakage controls

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