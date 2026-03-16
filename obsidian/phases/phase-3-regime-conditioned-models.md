# Phase 3 — Regime-Conditioned Models

**Status:** ⏳ Not started (blocked on Phase 2)

## Goal
Train predictive models conditioned on HMM regime labels. Handle financial non-stationarity by fitting separate or gated models per regime.

## Design proposals
- (none yet — add links here as proposals are written)

## Open questions
- [ ] Which model class per regime? (linear, tree-based, neural?)
- [ ] How to handle regime uncertainty at inference time?
- [ ] Walk-forward CV scheme — how to respect regime boundaries?
- [ ] Feature set per regime or shared?

## Constraints
- `src/regime_ml/models/` does not exist yet — do not import
- All design changes require a written proposal before implementation
