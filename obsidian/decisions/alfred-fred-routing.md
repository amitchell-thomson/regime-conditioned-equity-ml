---
decision: Per-series ALFRED vs FRED routing
date: pre-2026-03-16
status: settled
---

# Decision: ALFRED vs FRED Per-Series Routing

## Decision

Series routing is controlled by `use_alfred: true/false` per series in `regime_universe.yaml`.

**ALFRED (point-in-time vintages):** economically revised series
- CFNAI, INDPRO, PCEPILFE, NFCI, ICSA, CPIAUCSL, UNRATE, PAYEMS

**FRED (final revised values):** market-quoted daily series
- VIXCLS, DGS10, T10Y2Y, T10Y3M, FEDFUNDS, BAA10Y

## Rationale

ALFRED stores data as it was published at each date — the "vintage". Using ALFRED prevents look-ahead from data revisions. For example, CFNAI for 2008-09 was significantly revised after the fact; a backtest using final-revised CFNAI would show stronger recession signals than were actually available at the time.

Market-quoted daily series (VIX, yields, spreads) are not economically revised — the 10Y yield on 2008-09-15 is what it was. ALFRED is expensive and incomplete for these series; FRED final values are correct.

**Why:** Prevents backtest overfitting from data revision look-ahead in macro fundamentals. Market data has no revision to worry about.

**How to apply:** When adding a new series, ask: does FRED revise the historical values after initial release? If yes → use ALFRED. If it's a market price/rate → use FRED.
