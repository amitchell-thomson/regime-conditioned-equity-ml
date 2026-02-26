from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats
from scipy.optimize import linear_sum_assignment
from sklearn.covariance import LedoitWolf

from regime_ml.data.macro import build_featuregroup_map
from regime_ml.utils.config import load_configs

logger = logging.getLogger(__name__)


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
        "tv_distance": tv_mean,  # horizon controlled by n_mix parameter
    }


def evaluate_macro_coherence(
    X: np.ndarray,
    proba: np.ndarray,
    feature_names: list[str] | None = None,
    featuregroup_map: dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Macro coherence metrics using regime probabilities.

    1. Mahalanobis distance between regime means and variance explained by regimes:
        - are regime "macro signatures" well separated in feature space?
        - to do this we compute the pairwise Mahalanobis distance between regime mean vectors

    2. Anova style variance explained by regimes:
        - how much of the feature variance is explained by regimes?
        - to do this we compute explained R^2 per feature, then average

    Args:
        proba: (T, K) regime probability matrix. Pass filter_proba() output for causal
               evaluation (OOS). Smoothed probabilities are acceptable for IS interpretation.
    """
    X = np.asarray(X, float)
    proba = np.asarray(proba, float)

    T, d = X.shape
    T2, K = proba.shape
    assert T == T2, "X and proba must align on time dimension"

    # --- Weighted regime means (K,d)
    Nk = proba.sum(axis=0)  # (K,)
    Nk = np.maximum(Nk, 1e-12)
    mu_k = (proba.T @ X) / Nk[:, None]  # (K,d)

    # --- (A) Mahalanobis separation between regime means
    # Use shrinkage covariance for stability; get precision (inverse covariance)
    lw = LedoitWolf().fit(X)
    precision = lw.precision_  # (d,d)

    # Pairwise distances (upper triangle)
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            diff = mu_k[i] - mu_k[j]  # (d,)
            dist2 = float(diff @ precision @ diff)
            dists.append(np.sqrt(max(dist2, 0.0)))

    dists = np.array(dists, dtype=float) if dists else np.array([0.0])

    # --- (B) ANOVA-style variance explained (R2 per feature)
    mu = X.mean(axis=0)  # (d,)
    total_var = np.var(X, axis=0, ddof=1)
    total_var = np.maximum(total_var, 1e-12)

    wk = Nk / Nk.sum()  # (K,)
    between_var = np.sum(wk[:, None] * (mu_k - mu[None, :]) ** 2, axis=0)  # (d,)
    r2 = between_var / total_var  # (d,)

    # Top features by R2
    idx = np.argsort(-r2)

    def fname(i: int) -> str:
        return feature_names[i] if feature_names is not None else f"f{i}"

    top_n = 5
    top_feats = [(fname(i), float(r2[i])) for i in idx[:top_n]]

    # Average R^2 score per group

    name_to_idx = (
        {name: i for i, name in enumerate(feature_names)}
        if feature_names is not None
        else {}
    )
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


def validate_gaussian_assumption(
    X: np.ndarray,
    regime_labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Per-regime per-feature diagnostics of the Gaussian emission assumption.

    Computes skewness, excess kurtosis, and Shapiro-Wilk test statistics for
    each (regime, feature) pair. Also returns QQ-quantile data for notebook
    visualisation.

    **Analysis only** — this function MUST NOT be called in the fitting path
    or in any logic that produces trading signals. Use smooth_proba()-derived
    hard labels for cleanest regime separation when calling this function.

    Args:
        X:             (T, d) feature array (scaled, post-transform).
        regime_labels: (T,) integer hard-label array (e.g. from predict()).
        feature_names: Optional list of d feature names. Defaults to 'f0','f1',...

    Returns:
        Nested dict keyed by regime_id (int):
        {
            regime_id: {
                feature_name: {
                    "n":              int,          # observations in this regime
                    "skewness":       float,        # Fisher skewness
                    "kurtosis":       float,        # excess kurtosis (Gaussian = 0)
                    "shapiro_W":      float,        # Shapiro-Wilk W statistic
                    "shapiro_p":      float,        # p-value (low → non-Gaussian)
                    "qq_theoretical": np.ndarray,   # theoretical normal quantiles
                    "qq_sample":      np.ndarray,   # sample quantiles
                }
            }
        }

    Notes:
        - Shapiro-Wilk is set to NaN for regimes with < 8 observations (scipy
          minimum is 3, but 8 is the practical minimum for interpretable results).
        - Shapiro-Wilk is unreliable for n > 5000; the statistic is still
          returned but interpret with caution.
        - NaN and Inf values in X are dropped before computing statistics.
    """
    X = np.asarray(X, float)
    regime_labels = np.asarray(regime_labels, int)
    _, d = X.shape

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(d)]

    _SHAPIRO_MIN_N = 8
    results: dict = {}

    for regime_id in np.unique(regime_labels):
        mask = regime_labels == regime_id
        X_regime = X[mask]
        n_regime = X_regime.shape[0]
        regime_stats: dict = {}

        for j, fname in enumerate(feature_names):
            col = X_regime[:, j]
            col_clean = col[np.isfinite(col)]
            n_clean = len(col_clean)

            skew = float(scipy_stats.skew(col_clean)) if n_clean >= 2 else np.nan
            kurt = (
                float(scipy_stats.kurtosis(col_clean, fisher=True))
                if n_clean >= 2
                else np.nan
            )

            if n_clean >= _SHAPIRO_MIN_N:
                w_stat, p_val = scipy_stats.shapiro(col_clean)
                sw_w, sw_p = float(w_stat), float(p_val)
            else:
                sw_w, sw_p = np.nan, np.nan

            if n_clean >= 2:
                (osm, osr), _ = scipy_stats.probplot(col_clean, dist="norm")
                qq_theoretical = np.asarray(osm, float)
                qq_sample = np.asarray(osr, float)
            else:
                qq_theoretical = np.array([], float)
                qq_sample = np.array([], float)

            regime_stats[fname] = {
                "n": n_regime,
                "skewness": skew,
                "kurtosis": kurt,
                "shapiro_W": sw_w,
                "shapiro_p": sw_p,
                "qq_theoretical": qq_theoretical,
                "qq_sample": qq_sample,
            }

        results[int(regime_id)] = regime_stats

    return results


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
          ### optional override:
          "split_date": "YYYY-MM-DD"
      }
    """

    # Read n_mix horizon from config (evaluation.transmat_sanity.n_mix).
    _regime_cfg = load_configs().get("regimes", {})
    n_mix: int = int(
        _regime_cfg.get("evaluation", {}).get("transmat_sanity", {}).get("n_mix", 20)
    )

    # Build global group map once (for all possible features)
    global_group_map = build_featuregroup_map(list(features.columns))

    results: Dict[str, Any] = {}

    for model_id, model_data in models.items():
        model = model_data["model"]
        n_features = model_data["n_features"]
        scaler = model_data["scaler"]
        split_date = model_data["split_date"]
        this_split = pd.Timestamp(model_data.get("split_date", split_date))

        # Select the first n_features columns — features DataFrame is expected to have
        # PC columns ordered by explained variance (output of GroupPCATransformer.transform())
        selected_features = list(features.columns[:n_features])
        df_selected = features[selected_features]

        # Build group map only for selected features
        featuregroup_map = {
            f: global_group_map.get(f, "unknown") for f in selected_features
        }

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
        regimes_is = (
            model.predict(X_is) if X_is.shape[0] > 0 else np.array([], dtype=int)
        )
        regimes_oos = (
            model.predict(X_oos) if X_oos.shape[0] > 0 else np.array([], dtype=int)
        )

        smooth_is = model.smooth_proba(X_is) if X_is.shape[0] > 0 else None

        filt_is = model.filter_proba(X_is) if X_is.shape[0] > 0 else None
        filt_oos = model.filter_proba(X_oos) if X_oos.shape[0] > 0 else None

        # --- Metrics (FULL)
        regime_stability_full = evaluate_regime_stability(regimes_full)
        entropy_balance_full = evaluate_entropy_balance(filt_full)
        trans_sanity = evaluate_transmat_sanity(
            model.get_transition_matrix(), n_mix=n_mix
        )
        macro_coh_full = evaluate_macro_coherence(
            X_full_scaled,
            smooth_full,
            feature_names=selected_features,
            featuregroup_map=featuregroup_map,
        )

        # --- Metrics (IS / OOS)
        regime_stability_is = (
            evaluate_regime_stability(regimes_is) if regimes_is.size else {}
        )
        regime_stability_oos = (
            evaluate_regime_stability(regimes_oos) if regimes_oos.size else {}
        )

        entropy_balance_is = (
            evaluate_entropy_balance(filt_is) if filt_is is not None else {}
        )
        entropy_balance_oos = (
            evaluate_entropy_balance(filt_oos) if filt_oos is not None else {}
        )

        # IS: smoothed probabilities are acceptable for interpretation (not a trading signal).
        macro_coh_is = (
            evaluate_macro_coherence(
                X_is, smooth_is, selected_features, featuregroup_map
            )
            if smooth_is is not None
            else {}
        )
        # OOS: must use filter_proba (causal). smooth_proba runs forward-backward within the
        # OOS window, giving each time-step t access to t+1…T_oos — look-ahead bias.
        macro_coh_oos = (
            evaluate_macro_coherence(
                X_oos, filt_oos, selected_features, featuregroup_map
            )
            if filt_oos is not None
            else {}
        )

        # BIC / AIC — use scaled data for consistency with the model's training data
        bic_is = model.bic(X_is) if X_is.shape[0] > 0 else np.nan
        aic_is = model.aic(X_is) if X_is.shape[0] > 0 else np.nan
        bic_full = model.bic(X_full_scaled)
        aic_full = model.aic(X_full_scaled)

        results[model_id] = {
            "n_obs_full": int(X_full_scaled.shape[0]),
            "n_obs_is": int(X_is.shape[0]),
            "n_obs_oos": int(X_oos.shape[0]),
            # Complexity penalties
            "bic_is": bic_is,
            "aic_is": aic_is,
            "bic_full": bic_full,
            "aic_full": aic_full,
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
        sharpe = (
            float((ex_daily / vol_daily) * np.sqrt(freq)) if vol_daily > 0 else np.nan
        )

        # max drawdown within that regime's subsequence (contiguous in time is not required for this simple view)
        wealth = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(wealth)
        mdd = float(np.min(wealth / peak - 1.0))

        out.append(
            {
                "regime": r,
                "n_days": n,
                "mean_daily_ret": mean_daily,
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": mdd,
                "up_day_frac": float(np.mean(rets > 0)),
            }
        )

    return (
        pd.DataFrame(out)
        .sort_values("ann_return", ascending=False)
        .reset_index(drop=True)
    )


def expanding_window_cv(
    features_df: pd.DataFrame,
    model_config: Dict[str, Any],
    hmm_config: Dict[str, Any],
    cv_config: Dict[str, Any],
    train_end_date: pd.Timestamp,
) -> Dict[str, Any]:
    """Expanding-window cross-validation stability diagnostic.

    Refits the HMM with the given model_config on progressively longer IS
    windows and evaluates each model on a fixed-width OOS fold. Measures how
    consistent regime assignments are across folds (label churn) and how
    stable the OOS macro-coherence metric is (metric_cv_std).

    State alignment across folds uses Euclidean distance between regime means
    (compatible with all covariance types).

    DIAGNOSTIC ONLY — does not affect model selection. Results are stored in
    run_metadata.json for post-run inspection.

    Args:
        features_df:     Full feature DataFrame with DatetimeIndex (PC columns).
        model_config:    Dict with keys: n_regimes, covariance_type, p_stay.
        hmm_config:      Dict with keys: n_init, min_regime_share (from regime_config).
        cv_config:       Dict with keys: min_train_years, fold_step_months,
                         oos_window_months, max_churn_warning.
        train_end_date:  IS boundary — folds do not extend beyond this date.

    Returns:
        Dict with keys:
          label_churn     float — fraction of overlapping dates with label change
          metric_cv_std   float — std of OOS ANOVA-R² across folds
          n_folds         int   — number of complete folds evaluated
          fold_scores     list[float] — per-fold OOS ANOVA-R²
          churn_warning   bool  — True if label_churn > max_churn_warning
    """
    # Lazy import to avoid circular dependency at module level
    from regime_ml.regimes.hmm import fit_best_of_n_seeds  # noqa: PLC0415

    min_train_years = int(cv_config.get("min_train_years", 5))
    fold_step_months = int(cv_config.get("fold_step_months", 12))
    oos_window_months = int(cv_config.get("oos_window_months", 12))
    max_churn_warning = float(cv_config.get("max_churn_warning", 0.20))

    n_regimes = int(model_config["n_regimes"])
    covariance_type = str(model_config.get("covariance_type", "diag"))
    p_stay = float(model_config.get("p_stay", 0.95))
    n_init = int(hmm_config.get("n_init", 10))
    min_regime_share = float(hmm_config.get("min_regime_share", 0.03))

    feature_names = list(features_df.columns)
    featuregroup_map = build_featuregroup_map(feature_names)

    # Build fold IS-end dates: from min_train_years after data start up to train_end_date
    data_start = features_df.index[0]
    first_is_end = data_start + pd.DateOffset(years=min_train_years)
    fold_ends: list[pd.Timestamp] = []
    current = first_is_end
    while current <= train_end_date:
        fold_ends.append(current)
        current = current + pd.DateOffset(months=fold_step_months)

    if len(fold_ends) < 2:
        logger.warning(
            "expanding_window_cv: fewer than 2 folds available — "
            "increase data history or reduce min_train_years."
        )
        return {
            "label_churn": np.nan,
            "metric_cv_std": np.nan,
            "n_folds": 0,
            "fold_scores": [],
            "churn_warning": False,
        }

    fold_scores: list[float] = []
    fold_labels: list[np.ndarray] = []  # Viterbi labels per fold on the OOS window
    ref_detector = None

    for fold_end in fold_ends:
        oos_end = fold_end + pd.DateOffset(months=oos_window_months)
        df_is = features_df[features_df.index <= fold_end]
        df_oos = features_df[
            (features_df.index > fold_end) & (features_df.index <= oos_end)
        ]

        if len(df_is) < 30 or len(df_oos) < 10:
            logger.debug(
                "expanding_window_cv: skipping fold %s — insufficient data.",
                fold_end.date(),
            )
            continue

        try:
            detector, scaler = fit_best_of_n_seeds(
                df_is,
                n_regimes=n_regimes,
                n_init=n_init,
                train_end_date=fold_end,
                covariance_type=covariance_type,
                p_stay=p_stay,
                min_regime_share=min_regime_share,
                feature_names=feature_names,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "expanding_window_cv: fold %s fit failed: %s", fold_end.date(), exc
            )
            continue

        X_oos = scaler.transform(df_oos.values)

        # Align states to reference fold using Euclidean distance on means
        means_cand = detector.model.means_  # (K, d)
        if ref_detector is None:
            ref_detector = detector
            perm = np.arange(n_regimes)
        else:
            means_ref = ref_detector.model.means_
            cost = np.array(
                [
                    [
                        np.linalg.norm(means_cand[i] - means_ref[j])
                        for j in range(n_regimes)
                    ]
                    for i in range(n_regimes)
                ]
            )
            _, col_ind = linear_sum_assignment(cost)
            perm = col_ind

        raw_labels = detector.predict(X_oos)
        aligned_labels = perm[raw_labels]  # remap to reference ordering
        fold_labels.append(aligned_labels)

        # OOS macro coherence
        filt_oos = detector.filter_proba(X_oos)
        coh = evaluate_macro_coherence(
            X_oos,
            filt_oos,
            feature_names=feature_names,
            featuregroup_map=featuregroup_map,
        )
        fold_scores.append(float(coh.get("anova_r2_mean", np.nan)))

    n_folds = len(fold_scores)
    if n_folds == 0:
        return {
            "label_churn": np.nan,
            "metric_cv_std": np.nan,
            "n_folds": 0,
            "fold_scores": [],
            "churn_warning": False,
        }

    # Label churn: fraction of overlapping OOS dates where the label changed
    # Compare each consecutive fold pair on the common date range
    churn_fracs: list[float] = []
    for i in range(1, len(fold_labels)):
        a, b = fold_labels[i - 1], fold_labels[i]
        common_len = min(len(a), len(b))
        if common_len > 0:
            churn_fracs.append(float(np.mean(a[:common_len] != b[:common_len])))

    label_churn = float(np.mean(churn_fracs)) if churn_fracs else np.nan
    valid_scores = [s for s in fold_scores if np.isfinite(s)]
    metric_cv_std = float(np.std(valid_scores)) if len(valid_scores) > 1 else np.nan
    churn_warning = bool(np.isfinite(label_churn) and label_churn > max_churn_warning)

    if churn_warning:
        logger.warning(
            "expanding_window_cv: label_churn=%.3f exceeds threshold %.2f. "
            "Regime assignments are unstable across window sizes — "
            "consider adjusting n_regimes or p_stay.",
            label_churn,
            max_churn_warning,
        )

    return {
        "label_churn": round(label_churn, 4) if np.isfinite(label_churn) else None,
        "metric_cv_std": (
            round(metric_cv_std, 4) if np.isfinite(metric_cv_std) else None
        ),
        "n_folds": n_folds,
        "fold_scores": [round(s, 4) if np.isfinite(s) else None for s in fold_scores],
        "churn_warning": churn_warning,
    }


_DEFAULT_EPISODES_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "configs/regimes/economic_episodes.yaml"
)


def validate_against_episodes(
    regimes: pd.Series,
    label_results: list[dict],
    episodes_path: str | Path | None = None,
) -> pd.DataFrame:
    """Validate regime labels against known economic episodes.

    ANALYSIS ONLY — never call in fitting or trading-signal paths.

    For each episode, counts what fraction of days fell in:
      (a) the state whose archetype_key matches expected_archetype (expected_pct)
      (b) the overall dominant state (dominant_pct)

    Args:
        regimes:       pd.Series with DatetimeIndex; integer state indices
                       (e.g. from model.predict()).
        label_results: List of dicts from label_regimes() — must include
                       "state_idx", "archetype_key", "label".
        episodes_path: Optional override for configs/regimes/economic_episodes.yaml.

    Returns:
        DataFrame with one row per episode and columns:
          episode, start, end, expected_archetype,
          dominant_state, dominant_label, archetype_match,
          expected_pct, dominant_pct, n_days
    """
    ep_path = (
        Path(episodes_path) if episodes_path is not None else _DEFAULT_EPISODES_PATH
    )
    if not ep_path.exists():
        raise FileNotFoundError(f"economic_episodes.yaml not found at {ep_path}.")

    with open(ep_path, "r") as fh:
        raw = yaml.safe_load(fh)
    episodes = raw.get("episodes", [])

    # Build lookup: archetype_key → state_idx
    key_to_state: dict[str | None, int] = {
        r["archetype_key"]: r["state_idx"]
        for r in label_results
        if r.get("archetype_key") is not None
    }
    state_to_label: dict[int, str] = {r["state_idx"]: r["label"] for r in label_results}

    rows: list[dict] = []
    for ep in episodes:
        name = str(ep["name"])
        start = pd.Timestamp(ep["start"])
        end = pd.Timestamp(ep["end"])
        expected_key = str(ep["expected_archetype"])

        ep_regimes = regimes[(regimes.index >= start) & (regimes.index <= end)]
        n_days = len(ep_regimes)

        if n_days == 0:
            logger.debug(
                "validate_against_episodes: episode '%s' has no regime data — skipping.",
                name,
            )
            rows.append(
                {
                    "episode": name,
                    "start": start,
                    "end": end,
                    "expected_archetype": expected_key,
                    "dominant_state": None,
                    "dominant_label": None,
                    "archetype_match": False,
                    "expected_pct": np.nan,
                    "dominant_pct": np.nan,
                    "n_days": 0,
                }
            )
            continue

        # Dominant state
        counts = ep_regimes.value_counts()
        dominant_state = int(counts.index[0])
        dominant_pct = float(counts.iloc[0] / n_days)
        dominant_label = state_to_label.get(dominant_state, f"State {dominant_state}")

        # Expected state
        expected_state = key_to_state.get(expected_key)
        if expected_state is not None:
            expected_count = int((ep_regimes == expected_state).sum())
            expected_pct = float(expected_count / n_days)
        else:
            expected_pct = np.nan

        archetype_match = (
            dominant_state == expected_state if expected_state is not None else False
        )

        rows.append(
            {
                "episode": name,
                "start": start,
                "end": end,
                "expected_archetype": expected_key,
                "dominant_state": dominant_state,
                "dominant_label": dominant_label,
                "archetype_match": archetype_match,
                "expected_pct": (
                    round(expected_pct, 3) if np.isfinite(expected_pct) else np.nan
                ),
                "dominant_pct": round(dominant_pct, 3),
                "n_days": n_days,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        n_matched = int(df["archetype_match"].sum())
        logger.info(
            "validate_against_episodes: %d/%d episodes matched expected archetype.",
            n_matched,
            len(df),
        )
    return df
