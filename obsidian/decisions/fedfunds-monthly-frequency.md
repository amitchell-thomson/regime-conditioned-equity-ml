# FEDFUNDS Corrected to Monthly Frequency

**Decision:** FEDFUNDS series is monthly, not daily. Config corrected to reflect this.

**Why:** Treating a monthly series as daily causes incorrect staleness handling and inflated observation counts in rolling windows. The fix ensures the forward-fill and `is_new_data` logic handles it as a low-frequency series.

**Commit:** c726f35
