# CLAUDE.md

## Project

Regime-conditioned equity ML trading system. HMM-detected market regimes condition downstream predictive models to handle financial non-stationarity.

**Pipeline:** Raw Data → Load → Select → Clean → Align → Features → Regime Detection → Models → Portfolio

**Phase status:**
- ✅ Phases 1–2: Data pipeline, regime detection
- 🔄 Phases 3–4: Regime-conditioned modeling, portfolio evaluation — in progress

**NOT IMPLEMENTED — do not import, reference, or stub:**
- `src/regime_ml/models/` (Phase 3)
- `src/regime_ml/portfolio/` (Phase 4)

---

## Design Philosophy

These are the judgment calls that matter on this project. When in doubt, resolve ambiguity using these principles.

**Interpretability over complexity.** A feature with a clear economic mechanism (e.g. yield curve inversion signals credit stress) is preferred over one that improves in-sample fit without a causal story. If you can't explain why a feature should predict returns in a given regime, it shouldn't be in the model.

**Causal discipline is non-negotiable.** Every signal used in a trading decision must be constructable in real time with no future information. If a design requires non-causal data to work well, the design is wrong — not the constraint.

**Regime stability is a feature quality signal.** If a small hyperparameter change causes regime labels to flip materially, treat this as model instability, not acceptable variance. Features and models should produce coherent regime assignments across nearby configurations.

**Prefer economically sparse feature sets.** More features are not better. Redundant or highly correlated features (>0.8 pairwise correlation without a regime-conditional justification) should be excluded. The 5-indicator set exists for a reason.

**Simple and correct beats complex and leaky.** When a simpler approach and a more complex approach both solve a problem, prefer simple unless the complexity buys something measurable and leak-free out-of-sample.

---

## Hard Constraints

Violating these is always wrong regardless of context.

1. **Never compute rolling statistics on forward-filled data.** Low-frequency series (monthly, weekly) are forward-filled to daily. `is_new_data` tracks real observations. Transforms default to `staleness_mode='strict'` — compute on real data only, then forward-fill results. Explicitly set `staleness_mode='allow'` only when you have a specific reason.

2. **Never use `smooth_proba()` in trading logic.** `smooth_proba()` is non-causal (uses full history) — analysis and notebooks only. `filter_proba()` is causal — all trading signals. Mixing these introduces look-ahead bias that won't show up until live trading.

3. **Never hardcode parameters.** Thresholds, window lengths, and hyperparameters live in YAML configs under `configs/`. If you're writing a number directly into code, stop.

4. **Never mix in-sample and out-of-sample logic.** Be explicit about which side of the boundary you're on. Normalisation, fitting, and threshold selection must use only in-sample data.

---

## Before Implementing

- Does this touch regime inference or evaluation? → **Write a design proposal only. Do not implement.**
- Does this touch feature construction? → Verify staleness handling and check for look-ahead bias.
- Am I adding a parameter? → It goes in YAML.
- Am I creating a new file? → Check module ownership below first.

For large tasks: output a design proposal covering (1) approach, (2) integration points, (3) leakage/causality risks. Stop there. Do not implement until confirmed.

---

## Module Ownership

| Adding | Where |
|---|---|
| New transform | `features/common/transforms/` |
| New regime metric | `regimes/evaluation.py` |
| New selection criterion | `regimes/selection.py` |
| New label logic | `regimes/labeling.py` |
| New macro feature step | `features/macro/` |
| New config parameter | `configs/` YAML — never hardcoded |
| New macro data loader/cleaner | `data/macro/` |

---

## Architecture

**Key modules:**
- `data/macro/` — Load FRED parquet, select series, clean, align to business day calendar with staleness tracking
- `features/common/transforms/` — `BaseTransform`, `TransformRegistry`, `ChainedTransform`. Transforms declared in YAML
- `features/macro/` — Apply transform chains, 11-point feature validation, top-N selection
- `regimes/hmm.py` — HMM detector, KMeans initialisation, `smooth_proba()` / `filter_proba()`
- `regimes/evaluation.py` — Evaluation metrics
- `regimes/selection.py` — Two-stage selection: hard filters → soft weighted ranking
- `regimes/labeling.py` — Interpretable regime labeling
- `utils/config.py` — `load_configs()` centralised YAML loader

**Transform chain (YAML):**
```yaml
vix:
  transforms:
    - [level, {z_score: {window: 63}}]
    - [{diff: {periods: 5}}, {z_score: {window: 126}}]
```

**Feature naming:** `{INDICATOR}_{transform_chain}` e.g. `VIXCLS_diff_5_zscore_126`

**Data:** 5 indicators (T10Y3M, VIXCLS, NFCI, PCEPILFE, CFNAI), 2005–2026, parquet in `data/` (gitignored)

---

## Commands

```bash
uv sync
pytest tests/ -v
pytest -k "test_transform" -v
black src/ tests/
ruff check src/ tests/
ruff check --fix src/ tests/
```

---

## Code Standards

- Type hints on all functions
- `logging` not `print`
- Informative exception messages
- Functions under ~50 lines
- Vectorised pandas/numpy — no row loops
- Descriptive names — no `x`, `df2`, `tmp`
- Pure logic separated from I/O

**Tests:** Every new transform needs unit tests covering normal operation, NaNs, insufficient window length, and staleness behaviour. Run `pytest tests/ -v` before marking anything done.

---

## Anti-Patterns

- Rolling stats on forward-filled data without `staleness_mode='allow'`
- `smooth_proba()` in anything feeding a trading signal
- Hardcoded thresholds or window lengths
- Monolithic functions wrapping multiple pipeline stages
- Logic in notebooks that belongs in `src/`
- Imports from `models/` or `portfolio/` (not implemented)