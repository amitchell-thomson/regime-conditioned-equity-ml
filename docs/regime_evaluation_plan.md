# Model Evaluation and Selection Plan

This document outlines the systematic approach to evaluating and selecting Hidden Markov Models for regime detection. The goal is to identify models that are both statistically sound and practically useful for downstream equity prediction.

---

## 1. Hyperparameter Tuning Strategy

### Core Hyperparameters to Search

These materially change regime structure and downstream usefulness:

| Category | Hyperparameters | Options |
|----------|-----------------|---------|
| Regime structure | `n_regimes` | {2, 3, 4, 5} |
| Emissions | `covariance_type` | {'diag', 'full'} |
| Feature space | Feature subsets | Optional |
| Transition persistence | Diagonal weight in initial transmat | e.g. 0.9, 0.95, 0.99 |
| Training window | Date range | e.g. 2005–2018, 2005–2020 |
| Refit policy | Model update strategy | Fixed θ vs rolling refit|

---

## 2. Two-Stage Evaluation Philosophy

The evaluation answers two different questions:

### Stage A: "Is this a good regime model?"

Purely macro-level, structure-focused evaluation. Uses smoothed and filtered probabilities for diagnostics and realism checks.

### Stage B: "Is this regime model useful for equities?"

Downstream, predictive, causal evaluation. Uses only filtered probabilities and forward forecasts.

> **Critical Rule:** Models must pass Stage A before moving on to Stage B. Otherwise poorly specified equity regime models risk overfitting to noise.

---

## 3. Stage A: Macro-Regime Quality Metrics

These metrics are computed using:
- **Smoothed probabilities** → diagnostics only
- **Filtered probabilities** → realism / causality checks

### A1. Regime Persistence (Anti-Churn)

Key statistics:
- Mean regime duration
- Distribution of durations (not just mean)
- Excessive switching penalty

**Target behavior:**
- Not too jumpy (frequent regime changes)
- Not one regime dominating forever

### A2. Regime Entropy Balance

From filtered probabilities or hard labels:

$$H = -\sum_k p_k \log p_k$$

**Penalize:**
- Collapsed regimes
- Unused states

> Note: Do not require equal mass — macro regimes are inherently asymmetric.

### A3. Transition Matrix Sanity

Derived metrics:
- **Diagonal dominance** — persistence measure
- **Implied expected duration** — $\frac{1}{1-A_{kk}}$
- **Transition asymmetry** — e.g. crisis → recovery ≠ recovery → crisis

**Reject models where:**
- One regime is absorbing
- Transitions are unrealistically fast

### A4. Macro Coherence

Each regime should have distinct macro signatures. This ensures regimes are economically meaningful, not just statistical artifacts.

**Compute per-regime means** (using smoothed probabilities):
- Growth proxies
- Inflation proxies
- Volatility / stress proxies

**Measure separation:**
- Pairwise Mahalanobis distance between regimes
- ANOVA-style variance explained

### Stage A Output

For each model:
- A vector of macro-quality scores
- Hard filters to eliminate clearly bad models

**Only survivors proceed to Stage B.**

---

## 4. Stage B: Downstream Equity Usefulness

Evaluate using only causal objects:
- Filtered regime probabilities
- n-step-ahead forecasts from those probabilities

> **Critical Rule:** Never use smoothed probabilities in Stage B.

### B1. Conditional Return Separation

For each equity (or index):
- Bucket days by dominant filtered regime
- Compute per-regime:
  - Mean return
  - Volatility
  - Downside tail risk

**Score on:**
- Between-regime separation
- Stability across subperiods

This checks whether regimes correspond to different equity environments.

### B2. Regime-Conditioned Model Uplift (Gold Standard)

**This is the most important metric.**

For a simple baseline equity model (e.g. linear / tree):

1. **Train unconditional model:**
$$y_{t+1} \sim X_t$$

2. **Train regime-conditioned model:**
$$y_{t+1} \sim X_t \oplus P(z_t \mid x_{1:t})$$
or train separate models per regime.

**Compare:**
- Out-of-sample log-loss / MSE / Sharpe
- Across multiple assets
- Across multiple time splits

The incremental improvement is your signal. This directly answers: *"Does this regime model help trading models?"*

### B3. Horizon Alignment Test

Because you can forecast n-steps ahead, test conditioning on:
- $P(z_t)$
- $P(z_{t+5})$
- $P(z_{t+20})$

Some regimes matter with lead time (e.g. tightening cycles).

**Reward models where:** Forward regime forecasts improve equity predictions at realistic horizons.

### B4. Stability Across Refits

Refit the HMM on rolling windows and check:
- Regime identity stability
- Transition matrix drift
- Equity-model performance variance

**Penalize models whose usefulness is brittle to refitting.**

---

## 5. Scoring and Selection

### Step 1: Normalize Metrics

For each model `m`, compute:
- `macro_quality_score(m)`
- `equity_uplift_score(m)`
- `stability_score(m)`

Normalize each to [0, 1] across candidates.

### Step 2: Weighted Composite Score

$$\text{Score}(m) = 0.3 \cdot \text{MacroQuality} + 0.5 \cdot \text{EquityUplift} + 0.2 \cdot \text{Stability}$$

> Weights are defensible and can be tuned based on priorities.

### Step 3: Pareto Sanity Check

Before final selection:
- Ensure model is not dominated on all dimensions
- Avoid choosing a pathological "winner"

### Step 4: Select Champion + Challenger

Keep:
- **Best overall model** — highest composite score
- **1–2 challengers** — e.g. simpler or more stable alternatives

This is how professional systems manage regime uncertainty.

---

## 6. Final Pipeline Output

The evaluation pipeline should output:

| Output | Description |
|--------|-------------|
| `best_model_id` | Identifier of the selected model |
| Hyperparameters | Full configuration of the chosen model |
| Justification table | Macro metrics, equity uplift metrics, stability metrics |
| Saved probabilities | Filtered + forecast regime probabilities |

This makes the choice **auditable and defensible**.

---

## Summary

> **Tune HMM hyperparameters using macro-quality filters first, then select the final model based on incremental out-of-sample equity performance using causal (filtered + forward-forecasted) regime probabilities, aggregated via a composite score — not likelihood.**

---

## Next Steps

Potential extensions:
1. Formalize this into a single `evaluate_regime_model()` function
2. Build a minimal equity test harness (fast but informative)
3. Choose which assets / signals best stress-test regime usefulness
