# Phase 1 — Data Pipeline

**Status:** ✅ Complete

## What it does
- Loads FRED parquet data (`data/macro/`)
- Selects series, cleans, aligns to business day calendar
- Tracks staleness via `is_new_data` flag for low-frequency series (monthly/weekly forward-filled to daily)

## Key design decisions
- [[decisions/staleness-strict-default|Staleness strict mode as default]]
- Forward-fill only after transforms are applied on real observations

## Modules
- `data/macro/` — loaders, cleaners, alignment
- `utils/config.py` — centralised YAML loader
