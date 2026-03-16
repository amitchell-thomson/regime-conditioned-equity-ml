# Design: 5-State Surgical Split Initialisation

**Status:** Proposed — requires design approval before implementation
**Priority:** High — this is the primary path to resolving CV churn
**Module:** `regimes/hmm.py` (new function), `regimes/pipeline.py` (two-pass fitting)

---

## Problem

All five 5-state grid models currently fail the CV churn hard filter (>0.65). The 5-state EM landscape is multimodal — random KMeans seeds find different local optima, producing unstable solutions. More seeds (`n_init=20`) helps but is insufficient on its own.

The 4-state winner is structurally limited: it cannot separate mid-cycle expansion (2005-07) from Policy Tightening, and cannot represent late-cycle slowdown (2019) as a distinct regime. Both gaps produce collapsed CV folds.

## Proposed solution

Initialise the 5-state EM from the production 4-state model by splitting its most ambiguous state. This anchors 4 of the 5 new states to a known stable solution and dramatically narrows the EM search space.

## New function: `initialise_emissions_from_split()`

```python
def initialise_emissions_from_split(
    df_train: pd.DataFrame,
    base_detector: HMMRegimeDetector,
    base_scaler: StandardScaler,
    split_state: int,
    n_clusters_for_split: int = 2,
    random_state: int = 0,
    train_end_date: Optional[pd.Timestamp] = None,
    covariance_type: str = "full",
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Initialise a K+1 state HMM from a fitted K-state model by splitting one state."""
```

**Algorithm:**
1. Predict Viterbi hard labels for all IS observations using the base (4-state) model
2. Extract all observations assigned to `split_state`
3. Run KMeans(n_clusters=2) on those observations to produce two sub-clusters
4. Compute Ledoit-Wolf covariance for each sub-cluster
5. Assemble K+1 means: `[base_means[:split_state], sub_mean_0, sub_mean_1, base_means[split_state+1:]]`
6. Return with the **same scaler** (scaler does not change — IS-only boundary is preserved)

**Which state to split?** State 2 (Inflation/Policy Stress in the 4-state model) is the primary candidate — it over-captures 2017 expansion and 2019 slowdown. Splitting it should produce `expansion` and `policy_constrained` (or `expansion` and `slowdown`) cleanly.

Try all 4 states (4 surgical seeds) rather than hardcoding — EM will converge to the right solution even from an off-target split if initialisation is close enough.

## Pipeline change: two-pass fitting

```
Pass 1: Fit all n=4 grid points (current code — KMeans init, n_init=10)
         → Select best n=4 model (becomes "base" for pass 2)

Pass 2: Fit all n=5 grid points using:
         (a) n_surgical_seeds=4: initialise_emissions_from_split(base_4state, split_state)
             for split_state in [0, 1, 2, 3]
         (b) Remaining seeds: standard KMeans init (n_init_n5=20 total)
         → Standard degeneracy + CV filters apply

Final: Compare best n=4 and best n=5 via select_best_hmm_model()
```

## Config additions required

```yaml
# regime_config.yaml
initialisation:
  use_surgical_split_for_n5: true
  n_surgical_seeds: 4         # Try splitting each of the 4 base states
  n_init_n5: 20               # Total seeds for n=5 (more than n=4 due to multimodal landscape)

hmm:
  n_init: 10                  # Default for n=4
  # n_init_n5 above overrides for n=5 specifically
```

## Integration points

- `regimes/hmm.py` — add `initialise_emissions_from_split()`
- `regimes/pipeline.py` — add two-pass logic: fit n=4 first, extract best, pass to n=5 fitting
- `configs/regimes/regime_config.yaml` — add `initialisation` block

## Causality / leakage risks

- The surgical split uses Viterbi labels from the IS data — no OOS data involved ✅
- The scaler from Pass 1 (IS-only) is reused unchanged in Pass 2 ✅
- KMeans sub-clustering is on IS data only ✅
- The IS boundary `train_end_date` must be passed explicitly to prevent accidental OOS use ✅

## Expected outcome

- At least one n=5 grid model passes CV churn (< 0.65, target < 0.50)
- Collapsed folds 6, 7, 19 disappear: expansion (2005-07) maps to `expansion` state cleanly; slowdown (2019) maps to `slowdown` state cleanly
- Fold 1 (dot-com 2000) may remain a structural first-fold artifact — acceptable

## Module ownership

New function → `regimes/hmm.py`
Pipeline modification → `regimes/pipeline.py`
New config block → `configs/regimes/regime_config.yaml`

**Do not implement until this design is confirmed.**
