import numpy as np
import pandas as pd
from typing import Dict, Any
from regime_ml.data.macro import build_featuregroup_map
# uses your build_featuregroup_map(all_feature_names)

def label_regimes(
    X: np.ndarray,
    proba: np.ndarray,
    feature_names: list[str],
) -> Dict[str, Any]:
    """
    Label HMM regimes (state indices) using macro group signatures.

    Args:
        X: (T, d) feature matrix (ideally standardized)
        proba: (T, K) regime probabilities (smoothed preferred for interpretation)
        feature_names: list of length d, names corresponding to columns in X
        featuregroup_map: {feature_name: group_name} e.g. output of build_featuregroup_map(feature_names)
        label_set: which deterministic label schema to use

    Returns:
        Dict containing:
          - state_labels: {k: "Label"}
          - state_group_scores: {k: {group: score}}
          - state_feature_means: (K,d) list form for JSON friendliness
          - group_ordering: helpful ranks used in labeling
    """
    featuregroup_map = build_featuregroup_map(feature_names)

    X = np.asarray(X, float)
    gamma = np.asarray(proba, float)

    T, d = X.shape
    T2, K = gamma.shape
    assert T == T2, "X and proba must align on time dimension"
    assert len(feature_names) == d, "feature_names must match X columns"

    # --- Weighted regime means in feature space: mu_k (K,d)
    Nk = np.maximum(gamma.sum(axis=0), 1e-12)              # (K,)
    mu_k = (gamma.T @ X) / Nk[:, None]                     # (K,d)

    # --- Group aggregation: state_group_scores[k][group] = mean of mu_k over features in that group
    groups = sorted(set(featuregroup_map.get(f, "unknown") for f in feature_names))
    # build indices per group
    group_to_idx: Dict[str, list[int]] = {g: [] for g in groups}
    for j, f in enumerate(feature_names):
        g = featuregroup_map.get(f, "unknown")
        group_to_idx.setdefault(g, []).append(j)

    state_group_scores: Dict[int, Dict[str, float]] = {}
    for k in range(K):
        state_group_scores[k] = {}
        for g, idxs in group_to_idx.items():
            if len(idxs) == 0:
                continue
            state_group_scores[k][g] = float(np.mean(mu_k[k, idxs]))

    # Convenience arrays for ranking (missing groups -> 0.0)
    def gvec(gname: str) -> np.ndarray:
        return np.array([state_group_scores[k].get(gname, 0.0) for k in range(K)], dtype=float)

    growth = gvec("growth")
    inflation = gvec("inflation")
    rates = gvec("rates")
    liquidity = gvec("liquidity")
    stress = gvec("stress")  # might be "risk" or "volatility" in your config; see note below

    def zscore(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, float)
        s = v.std()
        if s < eps:
            return np.zeros_like(v)
        return (v - v.mean()) / s

    zg = zscore(growth)
    zi = zscore(inflation)
    zr = zscore(rates)
    zl = zscore(liquidity)
    zs = zscore(stress)

    labels = [
        ("Early Expansion / Liquidity Driven",          +1.3*zg + 1.1*zl - 1.4*zi - 1.0*zs - 0.3*zr),
        ("Recession / Risk-Off",                        -1.4*zg - 0.6*zl + 1.4*zs + 0.3*zi),
        ("Stagflation",                                 -1.1*zg + 1.5*zi + 0.6*zs + 0.3*zr),
        ("Policy-Contstrained Expansion",               +1.2*zi + 1.3*zr - 0.9*zl - 0.4*zg),
    ]

    # pick best label per regime
    state_labels = {}
    state_label_scores = {}
    for k in range(K):
        best_lab, best_score = None, -np.inf
        for lab, score_vec in labels:
            sc = float(score_vec[k])
            if sc > best_score:
                best_lab, best_score = lab, sc
        state_labels[k] = best_lab
        state_label_scores[k] = best_score
        

    # If multiple states got same label, keep them but you may want to disambiguate
    # deterministically by appending suffixes.
    counts = {}
    for k, lab in state_labels.items():
        counts[lab] = counts.get(lab, 0) + 1
    if any(v > 1 for v in counts.values()):
        seen = {}
        for k in sorted(state_labels):
            lab = state_labels[k]
            seen[lab] = seen.get(lab, 0) + 1
            if counts[lab] > 1:
                state_labels[k] = f"{lab} ({seen[lab]})"




    return {
        "state_labels": {int(k): v for k, v in state_labels.items()},
        "state_group_scores": {int(k): {g: float(v) for g, v in dct.items()} for k, dct in state_group_scores.items()},
        "state_feature_means": mu_k.tolist(),  # JSON friendly
        "group_ordering": {
            "growth_z_score": zg.tolist(),
            "inflation_z_score": zi.tolist(),
            "rates_z_score": zr.tolist(),
            "liquidity_z_score": zl.tolist(),
            "stress_z_score": zs.tolist(),
        },
        "groups_present": groups,
    }