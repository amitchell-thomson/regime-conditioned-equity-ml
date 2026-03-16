# Staleness Strict Mode as Default

**Decision:** Transforms default to `staleness_mode='strict'` — compute on real observations only, then forward-fill results.

**Why:** Low-frequency series (monthly, weekly) are forward-filled to daily for alignment. Computing rolling statistics on forward-filled values inflates effective sample size and introduces pseudo-precision on stale data. This is a subtle but serious form of look-ahead-adjacent bias.

**How to override:** Set `staleness_mode='allow'` explicitly when you have a documented reason (e.g. the transform is purely for display/analysis, not a trading signal).

**Enforced by:** `is_new_data` flag propagated through the data pipeline.
