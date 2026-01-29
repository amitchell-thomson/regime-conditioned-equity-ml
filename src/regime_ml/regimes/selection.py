import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

def _get(d: Dict[str, Any], path: str, default=np.nan):
    """Safe nested getter: path like 'macro_coherence.maha_median'."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _range_score(x: float, lo: float, hi: float, slack: float = 0.5) -> float:
    """
    Score in [0,1]. Best inside [lo,hi]. Linear decay outside.
    slack controls decay width as a fraction of band width.
    """
    if not np.isfinite(x):
        return 0.0
    width = hi - lo
    if width <= 0:
        return 0.0
    s = slack * width
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / max(s, 1e-12))
    return max(0.0, 1.0 - (x - hi) / max(s, 1e-12))

def select_best_hmm_model(
    results: Dict[str, Any],
    *,
    min_share: float = 0.03,
    max_share: float = 0.80,
    oos_min_share: float = 0.02,
    oos_max_share: float = 0.85,
    max_implied_duration: float = 3000.0,
    maha_min_quantile: float = 0.10,   # used to set a data-driven threshold
    top_n: int = 10
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      best_model_id,
      leaderboard_df (top_n survivors, with score breakdown),
      rejected_df (model_id + rejection reason)
    """
    rows = []
    for model_id, r in results.items():
        rows.append({
            "model_id": model_id,

            # --- FULL sample
            "tv_valid": bool(_get(r, "transition_matrix_sanity.tv_distance_valid", False)),
            "tv20": _get(r, "transition_matrix_sanity.tv_distance_20d"),
            "max_implied_duration": _get(r, "transition_matrix_sanity.max_implied_duration"),
            "median_implied_duration": _get(r, "transition_matrix_sanity.median_implied_duration"),
            "max_offdiag": _get(r, "transition_matrix_sanity.max_offdiag_transition"),

            "min_share": _get(r, "regime_stability.min_regime_share"),
            "max_share": _get(r, "regime_stability.max_regime_share"),
            "n_transitions": _get(r, "regime_stability.n_transitions"),
            "avg_persistence": _get(r, "regime_stability.avg_persistence"),

            "maha_min": _get(r, "macro_coherence.maha_min"),
            "maha_median": _get(r, "macro_coherence.maha_median"),
            "anova_r2_mean": _get(r, "macro_coherence.anova_r2_mean"),

            "entropy_mean": _get(r, "entropy_balance.entropy_balance"),

            # --- OOS slice (structural robustness)
            "oos_min_share": _get(r, "out_of_sample.regime_stability.min_regime_share"),
            "oos_max_share": _get(r, "out_of_sample.regime_stability.max_regime_share"),
            "oos_n_transitions": _get(r, "out_of_sample.regime_stability.n_transitions"),
            "oos_avg_persistence": _get(r, "out_of_sample.regime_stability.avg_persistence"),
            "oos_anova_r2_mean": _get(r, "out_of_sample.macro_coherence.anova_r2_mean"),
        })

    df = pd.DataFrame(rows)

    # Data-driven threshold for "redundant regimes"
    maha_thresh = float(df["maha_min"].quantile(maha_min_quantile))
    if not np.isfinite(maha_thresh):
        maha_thresh = 0.0

    # --- Hard rejection
    rej = []
    keep_mask = np.ones(len(df), dtype=bool)

    def reject(mask, reason):
        nonlocal keep_mask, rej
        idx = df.index[mask & keep_mask]
        for i in idx:
            rej.append({"model_id": df.loc[i, "model_id"], "reason": reason})
        keep_mask &= ~mask

    reject(~df["tv_valid"], "transition_matrix: stationary invalid (tv_distance_valid=False)")
    reject(df["min_share"] < min_share, f"dead regime (min_share < {min_share})")
    reject(df["max_share"] > max_share, f"collapsed regime (max_share > {max_share})")
    reject(df["max_implied_duration"] > max_implied_duration, f"absorbing regime (max_implied_duration > {max_implied_duration})")
    reject(df["maha_min"] < maha_thresh, f"redundant regimes (maha_min below {maha_min_quantile:.0%} quantile)")

    # OOS robustness (only apply if OOS metrics exist)
    if df["oos_min_share"].notna().any(): # type: ignore
        reject(df["oos_min_share"] < oos_min_share, f"OOS dead regime (oos_min_share < {oos_min_share})")
    if df["oos_max_share"].notna().any(): # type: ignore
        reject(df["oos_max_share"] > oos_max_share, f"OOS collapsed regime (oos_max_share > {oos_max_share})")

    survivors = df[keep_mask].copy()
    rejected_df = pd.DataFrame(rej).sort_values(["reason", "model_id"]) if rej else pd.DataFrame(columns=["model_id","reason"]) # type: ignore

    if survivors.empty:
        raise ValueError("No surviving models after degeneracy filters. Loosen thresholds or inspect why rejected_df is full.")

    # --- Scoring (simple, robust)
    # Normalize some "higher is better" metrics via ranks (robust to scale)
    def rrank(s, ascending=False):
        return s.rank(pct=True, ascending=ascending)

    # Macro score (higher better)
    macro = 0.5 * rrank(survivors["maha_median"], ascending=True) + 0.5 * rrank(survivors["anova_r2_mean"], ascending=True)

    # Transition realism: range preferences + penalty
    dur_score = survivors["median_implied_duration"].apply(lambda x: _range_score(x, 20, 200, slack=0.75)) # type: ignore
    tv_score = survivors["tv20"].apply(lambda x: _range_score(x, 0.05, 0.30, slack=1.0)) if survivors["tv20"].notna().any() else 0.0 # type: ignore
    off_pen  = rrank(survivors["max_offdiag"], ascending=False)  # lower offdiag => higher rank
    trans = 0.45 * dur_score + 0.35 * tv_score + 0.20 * off_pen

    # Stability/usability: less churn, decent persistence, not overly certain
    churn = rrank(survivors["n_transitions"], ascending=False)  # lower transitions => higher rank
    pers  = survivors["avg_persistence"].apply(lambda x: _range_score(x, 20, 200, slack=1.0)) # type: ignore
    ent   = rrank(survivors["entropy_mean"], ascending=True)    # higher entropy => higher rank
    stab = 0.45 * churn + 0.35 * pers + 0.20 * ent

    # OOS robustness penalty/bonus
    # Encourage macro coherence not collapsing OOS:
    if survivors["oos_anova_r2_mean"].notna().any(): # type: ignore
        oos_macro = rrank(survivors["oos_anova_r2_mean"], ascending=True)
    else:
        oos_macro = 0.0

    survivors["macro_score"] = macro
    survivors["transition_score"] = trans
    survivors["stability_score"] = stab
    survivors["oos_macro_score"] = oos_macro

    survivors["final_score"] = (
        0.40 * survivors["macro_score"] +
        0.30 * survivors["transition_score"] +
        0.25 * survivors["stability_score"] +
        0.05 * survivors["oos_macro_score"]
    )

    leaderboard = survivors.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True) # type: ignore
    best_model_id = str(leaderboard.loc[0, "model_id"])

    return best_model_id, leaderboard, rejected_df