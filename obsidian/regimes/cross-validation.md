# Cross-Validation Strategy

**Module:** `regimes/evaluation.py:561` — `expanding_window_cv()`
**Config:** `configs/regimes/regime_config.yaml` → `cross_validation` section

## Purpose

Detect HMM models that are unstable across the IS training period. A model that churns >65% of its historical labels as the IS window expands is disqualifying — the regime assignments are not repeatable.

## Algorithm

```
Train period: 2000-01-01 → 2019-01-01 (train_end_date)
min_train_years=8, fold_step=12mo → ~21 folds

Fold 0: train on 2000 → 2008
Fold 1: train on 2000 → 2009
...
Fold N: train on 2000 → 2019 (reference = full IS model)
```

**For each fold:**
1. Fit HMM on expanding IS data
2. Align state labels to **full-IS reference model** via Hungarian matching on centroids *(reference was changed from first-fold to full-IS model — more stable anchor)*
3. Measure fraction of IS history where fold labels ≠ reference labels = **label churn**

**Hard filter:** `label_churn > max_churn (0.65)` → model rejected regardless of other scores

## Config parameters

```yaml
cross_validation:
  enabled: true
  max_churn: 0.65          # Hard rejection threshold
  min_cv_folds: 4          # Minimum folds needed for a valid churn estimate
  fold_step: 12            # Months added per fold
  oos_window: 12           # OOS evaluation window per fold (months)
```

## Key design choices

- **Churn is a hard filter, not soft score** — redistributed its 0.15 weight to macro (+0.05), oos (+0.05), bic (+0.05). A churning model has unstable regime definitions.
- **Reference is full IS fit** — not the first/smallest fold. The first fold is the most unstable reference (smallest IS sample). Using the full-IS model means churn measures "divergence from the production model", which is what Phase 3 cares about.

## Root cause of current poor CV scores

The near-zero fold scores are a **structural 4-state limitation** — not a code bug. See [[known-issues#Issue 1]] for full diagnosis.

**Key finding from fold-date reconstruction:**

| Collapsed fold | IS-end | Root cause |
|---|---|---|
| ~Fold 1 | ~2000 | Dot-com bust onset — model never seen tech crash |
| ~Fold 6 | ~2005 | Mid-cycle 2005-07 expansion — ambiguous in 4-state |
| ~Fold 7 | ~2006 | Pre-GFC expansion — same ambiguity |
| ~Fold 19 | ~2018 | 2019 late-cycle slowdown — no clean 4-state match |

These 4 folds drive `label_churn=0.643` (4-state winner). Without them, estimated mean churn falls well below 0.30. **Fix: 5-state model, not threshold tuning.**

## Proposed additional filter: fold-collapse hard filter

After surgical 5-state split is in place, add to config:

```yaml
cross_validation:
  max_zero_folds: 2          # Reject if >2 folds have anova_r2 below threshold
  zero_fold_threshold: 0.05  # "Collapsed" fold definition
```

A model with >2 structurally collapsed folds has discriminative failures in ≥10% of IS history — unacceptable for Phase 3 conditioning.

## Proposed CV output enhancement

Add `fold_metadata` to CV return dict:
```python
{
    "fold_index": int,
    "is_end": str,        # e.g. "2005-01-01"
    "oos_start": str,
    "oos_end": str,
    "anova_r2": float,
    "is_near_zero": bool  # anova_r2 < zero_fold_threshold
}
```
Store in `run_metadata.json` under `cv_diagnostics.fold_metadata`. Enables direct inspection of which economic periods collapsed.

## Phase 3 churn target

After surgical split + PPIACO enrichment:
- Mean churn < 0.30
- Near-zero folds ≤ 1 of 21
- max_pairwise_churn < 0.60
- Tighten `max_churn` from 0.65 to 0.50 once 5-state model achieves these

## Relationship to pipeline

In `regimes/pipeline.py`:
1. All grid models evaluated (metrics only, no CV)
2. Models passing hard filters → run CV (saves compute on clearly bad models)
3. CV churn failures added to `churn_rejected_ids`
4. Final selection passes `churn_rejected_ids` as hard-reject set
5. If **all** models fail CV → fallback to pre-CV selection (warning logged)
