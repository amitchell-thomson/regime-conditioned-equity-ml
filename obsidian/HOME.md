# Regime-Conditioned Equity ML

Research vault for the regime-conditioned equity ML trading system.

## Pipeline
Raw Data → Load → Select → Clean → Align → Features → Regime Detection → Models → Portfolio

## Phase Status
- ✅ [[phases/phase-1-data-pipeline|Phase 1 — Data Pipeline]]
- 🔄 [[phases/phase-2-regime-detection|Phase 2 — Regime Detection]] (nearly done — 2 blockers)
- ⏳ [[phases/phase-3-regime-conditioned-models|Phase 3 — Regime-Conditioned Models]] (not started — blocked on Phase 2)
- ⏳ [[phases/phase-4-portfolio-evaluation|Phase 4 — Portfolio Evaluation]] (not started)

## Current state
→ [[context/now|Current context]] — where we are, active questions, settled decisions

## Key Areas
- [[features/README|Features]] — transform chains, indicator selection
- [[regimes/README|Regimes]] — HMM, labeling, evaluation, selection
- [[decisions/README|Decisions]] — architectural decisions and rationale
- [[designs/README|Design Proposals]] — proposals pending or approved

## Hard Constraints (from CLAUDE.md)
1. Never compute rolling stats on forward-filled data
2. Never use `smooth_proba()` in trading logic — use `filter_proba()`
3. Never hardcode parameters — YAML configs only
4. Never mix in-sample and out-of-sample logic
