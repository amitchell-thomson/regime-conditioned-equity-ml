# Regime Labeling

**Module:** `regimes/labeling.py`
**Config:** `configs/regimes/regime_archetypes.yaml`

## Purpose

Match HMM state centroids to named economic archetypes. Labels are **analysis-only** — never feed label assignments into trading signals; use `filter_proba()` for that.

## Algorithm

```
1. Load archetype pool (YAML), select pool by K:
      K ≤ 3  → n3 pool  (risk_on, contraction, crisis)
      K = 4  → n4 pool  (risk_on, contraction, inflation_policy, crisis)
      K ≥ 5  → canonical (expansion, recovery, slowdown, policy_constrained, liquidity_crisis)

2. Compute state vectors (K, 5 groups):
      state_vecs[k, g] = probability-weighted mean of PC1 for group g across all time steps
      Uses smooth_proba() weights for cleanest separation (analysis context)
      ** Uses PC1 only ** — PC2+ dilute the group signal

3. Cosine similarity: S[k, j] = cos(state_vecs[k], archetype_signatures[j])
   Both vectors are L2-normalised before dot product

4. Linear assignment (Hungarian): maximise sum(S[k, assigned(k)])
   No archetype assigned to more than one state

5. Confidence check:
      confidence = S[k, assigned(k)]
      if confidence < min_confidence (0.18) → "Unclassified Regime {k}"
      if margin < 0.08 → margin_warning=True (label still assigned)
```

## Archetype signature directions

All PC1 signs are anchored: **positive = economically good**

| Group | Positive means | Negative means |
|---|---|---|
| rates | Steep yield curve + low short rates (easing cycle) | Inverted curve + high short rates (tightening) |
| inflation | Above-target inflation | Disinflation / deflation |
| real_economy | Strong growth + tight labour market | Recession / labour market stress |
| credit | Easy financial conditions (low NFCI, tight spreads) | Financial stress (high NFCI, wide spreads) |
| volatility | Calm markets (low VIX) | Market stress (high VIX) |

## Canonical archetypes (K ≥ 5)

| Archetype | Key signature | Canonical periods |
|---|---|---|
| Expansion | real_economy +1.2, credit +1.0, vol +1.0 | 1995-99, 2003-07, 2013-18 |
| Recovery | real_economy +0.7, credit +0.6, rates +0.7 (steep curve) | 2009-10, 2020H2 |
| Late-Cycle Disinflation | inflation -1.0, rates -0.8, real_economy 0.0 | 2015-16, 2019, 2023 |
| Policy Tightening | rates -1.5, inflation +1.0, real_economy +0.5 | 2018, 2022-24 |
| Liquidity Crisis | credit -2.5, volatility -2.5, real_economy -1.0 | 2008-09 GFC, 2020 COVID |

## Output structure per state

```python
{
    "state_idx":         int,
    "label":             str,       # e.g. "Expansion" or "Unclassified Regime 2"
    "status":            str,       # "matched" | "unclassified"
    "confidence":        float,     # cosine similarity of best match (0 to 1)
    "margin":            float,     # best_score - runner_up_score
    "margin_warning":    bool,      # True if margin < 0.08
    "archetype_key":     str|None,
    "runner_up":         str,
    "runner_up_score":   float,
    "pool":              str,       # "n3" | "n4" | "canonical"
}
```

## Known issue: confidence scores low

See [[known-issues#Issue 2: Regime archetype confidence scores need fixing]]

Most likely cause: `featuregroup_map` fallback for PCA columns (`rates_pc1`, etc.) may not be mapping features to groups correctly, leaving zeros in state_vecs and degrading cosine similarity scores.

## featuregroup_map fallback

For PCA features (e.g. `rates_pc1`, `real_economy_pc1`), there is a fallback in `label_regimes()`:
```python
for feat in feature_names:
    if featuregroup_map.get(feat, "unknown") == "unknown":
        for g in groups_set:
            if feat.startswith(f"{g}_") or feat == g:
                featuregroup_map[feat] = g
                break
```
This is needed because `build_featuregroup_map()` maps FRED series codes (e.g. `VIXCLS`) not PCA column names. The fallback must fire correctly for all 5 groups.
