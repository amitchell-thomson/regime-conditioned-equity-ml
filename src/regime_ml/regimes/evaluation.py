import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.covariance import LedoitWolf
from regime_ml.data.macro import build_featuregroup_map
from regime_ml.features.macro.selection import get_top_features

def evaluate_regime_stability(regimes: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate regime stability metrics.
    
    Returns:
        - persistence: Average regime duration (days)
        - n_transitions: Number of regime changes
        - entropy: Regime distribution entropy (balanced = high)
    """
    regimes = np.asarray(regimes)

    transitions = np.diff(regimes)
    n_transitions = int(np.sum(transitions != 0))

    regime_lengths = []
    current_regime = regimes[0]
    current_length = 1
    for r in regimes[1:]:
        if r == current_regime:
            current_length += 1
        else:
            regime_lengths.append(current_length)
            current_regime = r
            current_length = 1
    regime_lengths.append(current_length)

    avg_persistence = float(np.mean(regime_lengths))
    std_persistence = float(np.std(regime_lengths))

    unique, counts = np.unique(regimes, return_counts=True)
    T = len(regimes)
    shares = counts / T

    min_regime_share = float(np.min(shares))
    max_regime_share = float(np.max(shares))

    entropy = float(-np.sum(shares * np.log2(np.clip(shares, 1e-12, 1.0))))

    return {
        "avg_persistence": avg_persistence,
        "std_persistence": std_persistence,
        "n_transitions": n_transitions,
        "regime_entropy": entropy,
        "regime_counts": dict(zip(unique, counts)),
        "min_regime_share": min_regime_share,
        "max_regime_share": max_regime_share,
    }

def evaluate_entropy_balance(proba: np.ndarray, eps: float = 1e-12) -> Dict[str, Any]:
    """
    Entropy of a probability vector (or matrix) with safe handling of zeros.

    If proba is (K,), returns entropy of that distribution.
    If proba is (T,K), returns mean entropy across time.
    """
    p = np.asarray(proba, float)

    # If matrix, compute per-row entropy then average
    if p.ndim == 2:
        p = np.clip(p, eps, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        H = -np.sum(p * np.log2(p), axis=1)
        return {
            "entropy_balance": float(np.mean(H)),
            "entropy_balance_std": float(np.std(H)),
        }

    # Vector case
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    H = -float(np.sum(p * np.log2(p)))
    return {"entropy_balance": H}

def evaluate_transmat_sanity(A: np.ndarray, n_mix: int = 20) -> Dict[str, Any]:
    """
    Evaluate sanity of an HMM transition matrix.

    Args:
        A: (K,K) transition matrix
        n_mix: horizon for mixing diagnostic (e.g. 20 trading days)

    Returns:
        Dictionary of high-signal transition metrics
    """
    A = np.asarray(A, float)
    diag = np.clip(np.diag(A), 0.0, 1.0 - 1e-12)
    durations = 1.0 / (1.0 - diag)

    def stationary_dist(A: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
        try:
            w, V = np.linalg.eig(A.T)
        except np.linalg.LinAlgError:
            return None
        idx = np.argmin(np.abs(w - 1.0))
        v = np.real(V[:, idx])
        v = np.maximum(v, 0.0)
        s = v.sum()
        if not np.isfinite(s) or s < eps:
            return None
        return v / s

    pi = stationary_dist(A)
    if pi is None:
        stationary_valid = False
        tv_mean = np.nan
    else:
        stationary_valid = True
        A_n = np.linalg.matrix_power(A, n_mix)
        tv = 0.5 * np.sum(np.abs(A_n - pi[None, :]), axis=1)  # (K,)
        tv_mean = float(np.mean(tv))

    P = np.clip(A, 1e-12, 1.0)
    row_entropy = -np.sum(P * np.log2(P), axis=1)

    off = A.copy()
    np.fill_diagonal(off, 0.0)

    return {
        "median_implied_duration": float(np.median(durations)),
        "max_implied_duration": float(np.max(durations)),
        "mean_self_transition": float(np.mean(diag)),
        "max_offdiag_transition": float(off.max()),
        "mean_row_entropy": float(np.mean(row_entropy)),
        "tv_distance_valid": bool(stationary_valid),
        f"tv_distance_{n_mix}d": tv_mean,
    }

def evaluate_macro_coherence(
    X: np.ndarray,
    smooth_proba: np.ndarray,
    feature_names: list[str] | None = None,
    featuregroup_map: dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Macro coherence metrics using smoothed regime probabilities.

    1. Mahalanobis distance between regime means and variance explained by regimes:
        - are regime "macro signatures" well separated in feature space?
        - to do this we compute the pairwise Mahalanobis distance between regime mean vectors

    2. Anova style variance explained by regimes:
        - how much of the feature variance is explained by regimes?
        - to do this we compute explained R^2 per feature, then average
    
    """
    X = np.asarray(X, float)
    smooth_proba = np.asarray(smooth_proba, float)

    T, d = X.shape
    T2, K = smooth_proba.shape
    assert T == T2, "X and smooth_proba must align on time dimension"

    # --- Weighted regime means (K,d)
    Nk = smooth_proba.sum(axis=0)                       # (K,)
    Nk = np.maximum(Nk, 1e-12)
    mu_k = (smooth_proba.T @ X) / Nk[:, None]           # (K,d)

    # --- (A) Mahalanobis separation between regime means
    # Use shrinkage covariance for stability; get precision (inverse covariance)
    lw = LedoitWolf().fit(X)
    precision = lw.precision_                    # (d,d)

    # Pairwise distances (upper triangle)
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            diff = mu_k[i] - mu_k[j]             # (d,)
            dist2 = float(diff @ precision @ diff)
            dists.append(np.sqrt(max(dist2, 0.0)))

    dists = np.array(dists, dtype=float) if dists else np.array([0.0])

    # --- (B) ANOVA-style variance explained (R2 per feature)
    mu = X.mean(axis=0)                          # (d,)
    total_var = np.var(X, axis=0, ddof=1)
    total_var = np.maximum(total_var, 1e-12)

    wk = Nk / Nk.sum()                           # (K,)
    between_var = np.sum(wk[:, None] * (mu_k - mu[None, :])**2, axis=0)  # (d,)
    r2 = between_var / total_var                 # (d,)

    # Top features by R2
    idx = np.argsort(-r2)
    def fname(i: int) -> str:
        return feature_names[i] if feature_names is not None else f"f{i}"
    top_n = 5
    top_feats = [(fname(i), float(r2[i])) for i in idx[:top_n]]

    # Average R^2 score per group

    name_to_idx = {name: i for i, name in enumerate(feature_names)} if feature_names is not None else {}
    group_r2: Dict[str, float] = {}
    if featuregroup_map is not None:
        # featuregroup_map: {feature_name: group_name}
        tmp: Dict[str, list[float]] = {}

        for feat, group in featuregroup_map.items():
            if feature_names is None:
                continue  # can't map names to indices
            if feat not in name_to_idx:
                continue  # skip unknown features

            tmp.setdefault(group, []).append(float(r2[name_to_idx[feat]]))

        group_r2 = {g: float(np.mean(vals)) for g, vals in tmp.items() if len(vals) > 0}

    return {
        # 1–3: regime separation in macro space
        "maha_min": float(np.min(dists)),
        "maha_median": float(np.median(dists)),
        "maha_mean": float(np.mean(dists)),

        # 4–5: how much regimes explain macro features
        "anova_r2_mean": float(np.mean(r2)),
        "anova_r2_median": float(np.median(r2)),

        # 6. Per group average R^2 score
        "anova_group_r2": group_r2,

        # 7: what features define regimes
        "anova_top_features": top_feats,
    }

def compare_hmm_models(
    features: pd.DataFrame,
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare multiple trained HMM models on full sample, and separately on IS vs OOS.

    models dict expected:
      models[model_id] = {
          "model": fitted_HMMRegimeDetector,
          "n_features": int,
          "scaler": fitted scaler,
          # optional override:
          "split_date": "YYYY-MM-DD"
      }
    """

    # Build global group map once (for all possible features)
    global_group_map = build_featuregroup_map(list(features.columns))

    results: Dict[str, Any] = {}

    for model_id, model_data in models.items():
        model = model_data["model"]
        n_features = model_data["n_features"]
        scaler = model_data["scaler"]
        split_date = model_data["split_date"]
        this_split = pd.Timestamp(model_data.get("split_date", split_date))

        # Select features (kept as your current approach)
        selected_features = get_top_features(n=n_features)
        df_selected = features[selected_features]

        # Build group map only for selected features
        featuregroup_map = {f: global_group_map.get(f, "unknown") for f in selected_features}

        # Scale
        X_full = df_selected.values
        X_full_scaled = scaler.transform(X_full)

        # Split by time index
        is_mask = df_selected.index <= this_split
        oos_mask = df_selected.index > this_split

        X_is = X_full_scaled[is_mask]
        X_oos = X_full_scaled[oos_mask]

        # --- FULL sample outputs (as before)
        regimes_full = model.predict(X_full_scaled)
        smooth_full = model.smooth_proba(X_full_scaled)
        filt_full = model.filter_proba(X_full_scaled)

        # --- IS / OOS outputs
        # (compute on each slice separately; avoids any accidental cross-slice dependence)
        regimes_is = model.predict(X_is) if X_is.shape[0] > 0 else np.array([], dtype=int)
        regimes_oos = model.predict(X_oos) if X_oos.shape[0] > 0 else np.array([], dtype=int)

        smooth_is = model.smooth_proba(X_is) if X_is.shape[0] > 0 else None
        smooth_oos = model.smooth_proba(X_oos) if X_oos.shape[0] > 0 else None

        filt_is = model.filter_proba(X_is) if X_is.shape[0] > 0 else None
        filt_oos = model.filter_proba(X_oos) if X_oos.shape[0] > 0 else None

        # --- Metrics (FULL)
        regime_stability_full = evaluate_regime_stability(regimes_full)
        entropy_balance_full = evaluate_entropy_balance(filt_full)
        trans_sanity = evaluate_transmat_sanity(model.get_transition_matrix(), n_mix=20)
        macro_coh_full = evaluate_macro_coherence(
            X_full_scaled, smooth_full,
            feature_names=selected_features,
            featuregroup_map=featuregroup_map
        )

        # --- Metrics (IS / OOS)
        regime_stability_is = evaluate_regime_stability(regimes_is) if regimes_is.size else {}
        regime_stability_oos = evaluate_regime_stability(regimes_oos) if regimes_oos.size else {}

        entropy_balance_is = evaluate_entropy_balance(filt_is) if filt_is is not None else {}
        entropy_balance_oos = evaluate_entropy_balance(filt_oos) if filt_oos is not None else {}

        macro_coh_is = evaluate_macro_coherence(X_is, smooth_is, selected_features, featuregroup_map) if smooth_is is not None else {}
        macro_coh_oos = evaluate_macro_coherence(X_oos, smooth_oos, selected_features, featuregroup_map) if smooth_oos is not None else {}

        results[model_id] = {
            "n_obs_full": int(X_full_scaled.shape[0]),
            "n_obs_is": int(X_is.shape[0]),
            "n_obs_oos": int(X_oos.shape[0]),

            # Full-sample (as before)
            "regime_stability": regime_stability_full,
            "entropy_balance": entropy_balance_full,
            "transition_matrix_sanity": trans_sanity,
            "macro_coherence": macro_coh_full,

            # New: IS vs OOS (structural validation)
            "in_sample": {
                "regime_stability": regime_stability_is,
                "entropy_balance": entropy_balance_is,
                "macro_coherence": macro_coh_is,
            },
            "out_of_sample": {
                "regime_stability": regime_stability_oos,
                "entropy_balance": entropy_balance_oos,
                "macro_coherence": macro_coh_oos,
            },
        }

    return results


def equity_metrics_by_regime(
    px: pd.Series,
    regimes: pd.Series | np.ndarray,
    *,
    freq: int = 252,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """
    Compute average equity metrics per regime.

    Args:
        px: price series indexed by date (e.g. SPY adj close)
        regimes: regime labels indexed by date (pd.Series) OR numpy array aligned to px index
        freq: trading days per year
        rf_annual: annual risk-free rate (e.g. 0.03 for 3%)

    Returns:
        DataFrame with metrics per regime.
    """
    # px = px.dropna().astype(float)

    if isinstance(regimes, pd.Series):
        df = pd.DataFrame({"px": px}).join(regimes.rename("regime"), how="inner")
    else:
        # assume regimes is aligned to px index
        df = pd.DataFrame({"px": px, "regime": np.asarray(regimes)}, index=px.index)

    df = df.dropna()
    df["ret"] = df["px"].pct_change()
    df = df.dropna(subset=["ret"])

    rf_daily = (1.0 + rf_annual) ** (1.0 / freq) - 1.0

    out = []
    for r, g in df.groupby("regime"):
        rets = g["ret"].to_numpy()
        n = len(rets)
        if n < 2:
            continue

        mean_daily = float(np.mean(rets))
        vol_daily = float(np.std(rets, ddof=1))
        ann_ret = float((1.0 + mean_daily) ** freq - 1.0)
        ann_vol = float(vol_daily * np.sqrt(freq))

        ex_daily = mean_daily - rf_daily
        sharpe = float((ex_daily / vol_daily) * np.sqrt(freq)) if vol_daily > 0 else np.nan

        # max drawdown within that regime's subsequence (contiguous in time is not required for this simple view)
        wealth = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(wealth)
        mdd = float(np.min(wealth / peak - 1.0))

        out.append({
            "regime": r,
            "n_days": n,
            "mean_daily_ret": mean_daily,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "up_day_frac": float(np.mean(rets > 0)),
        })

    return pd.DataFrame(out).sort_values("ann_return", ascending=False).reset_index(drop=True)