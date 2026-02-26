import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def _get(d: Dict[str, Any], path: str, default=np.nan):
    """Safe nested getter: path like 'macro_coherence.maha_median'."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _soft_score(
    x: float, optimal: float, lo: float, hi: float, slack: float = 0.5
) -> float:
    """Score in [0, 1]. Returns 1.0 at optimal; decays linearly outward.

    Within [lo, hi]: linear gradient from 0.5 at the boundaries to 1.0 at optimal.
    Outside [lo, hi]: decays linearly from 0.5 at the boundary to 0.0 at
    lo − slack·width (or hi + slack·width).

    This replaces the flat-top _range_score which gave 1.0 everywhere inside
    [lo, hi], making models within the range indistinguishable.

    Args:
        x:       Metric value to score.
        optimal: Ideal value — receives score 1.0.
        lo, hi:  Acceptable range boundaries — receive score 0.5.
        slack:   Decay width beyond boundaries as fraction of (hi - lo).
    """
    if not np.isfinite(x):
        return 0.0
    width = hi - lo
    if width <= 0:
        return 0.0
    opt = float(np.clip(optimal, lo, hi))
    s = slack * width
    if lo <= x <= hi:
        if x <= opt:
            span = opt - lo
            return float(0.5 + 0.5 * (x - lo) / span) if span > 1e-12 else 1.0
        else:
            span = hi - opt
            return float(0.5 + 0.5 * (hi - x) / span) if span > 1e-12 else 1.0
    elif x < lo:
        return float(max(0.0, 0.5 * (x - (lo - s)) / s)) if s > 1e-12 else 0.0
    else:  # x > hi
        return float(max(0.0, 0.5 * (hi + s - x) / s)) if s > 1e-12 else 0.0

def select_best_hmm_model(
    results: Dict[str, Any],
    *,
    min_share: float = 0.03,
    max_share: float = 0.80,
    oos_min_share: float = 0.02,
    oos_max_share: float = 0.85,
    max_implied_duration: float = 3000.0,
    maha_min_quantile: float = 0.10,
    top_n: int = 10,
    weights: Dict[str, float] | None = None,
    soft_score_cfg: Dict[str, Any] | None = None,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      best_model_id,
      leaderboard_df (top_n survivors, with score breakdown),
      rejected_df (model_id + rejection reason)
    """
    # --- Scoring weights (override via weights kwarg or use defaults)
    _default_weights = {
        "macro": 0.25,
        "transitions": 0.25,
        "stability": 0.20,
        "oos": 0.20,
        "bic": 0.10,
    }
    w = {**_default_weights, **(weights or {})}
    # Normalise in case caller passed unnormalised weights
    w_total = sum(w.values())
    w = {k: v / w_total for k, v in w.items()}

    rows = []
    for model_id, r in results.items():
        rows.append({
            "model_id": model_id,

            # --- FULL sample
            "tv_valid": bool(_get(r, "transition_matrix_sanity.tv_distance_valid", False)),
            "tv20": _get(r, "transition_matrix_sanity.tv_distance"),
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

            # --- BIC / AIC (complexity penalties)
            "bic_full": _get(r, "bic_full"),

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

    # A non-stationary transition matrix has no meaningful long-run distribution;
    # regime labels have no stable economic interpretation.
    reject(~df["tv_valid"], "transition_matrix: stationary invalid (tv_distance_valid=False)")
    # A regime with <3% share is economically inert — too rare to build a strategy on.
    # A regime with >80% share dominates the sample, implying near-trivial segmentation.
    reject(df["min_share"] < min_share, f"dead regime (min_share < {min_share})")
    reject(df["max_share"] > max_share, f"collapsed regime (max_share > {max_share})")
    # Implied duration >3000 days (~12 years) means the self-transition probability is
    # nearly 1 — the model cannot escape the regime; it is effectively absorbing.
    reject(df["max_implied_duration"] > max_implied_duration, f"absorbing regime (max_implied_duration > {max_implied_duration})")
    # Mahalanobis distance below the 10th percentile means at least two regimes are
    # indistinguishable in macro feature space — a degenerate, redundant solution.
    reject(df["maha_min"] < maha_thresh, f"redundant regimes (maha_min below {maha_min_quantile:.0%} quantile)")

    # OOS robustness: dead/collapsed regimes OOS signal overfitting to the training period.
    if df["oos_min_share"].notna().any():  # type: ignore[union-attr]  # pandas Series notna().any() — mypy cannot narrow after column indexing
        reject(df["oos_min_share"] < oos_min_share, f"OOS dead regime (oos_min_share < {oos_min_share})")
    if df["oos_max_share"].notna().any():  # type: ignore[union-attr]  # pandas Series notna().any() — mypy cannot narrow after column indexing
        reject(df["oos_max_share"] > oos_max_share, f"OOS collapsed regime (oos_max_share > {oos_max_share})")

    survivors = df[keep_mask].copy()
    rejected_df = pd.DataFrame(rej).sort_values(["reason", "model_id"]) if rej else pd.DataFrame(columns=["model_id", "reason"])  # type: ignore[arg-type]  # sort_values returns DataFrame but chain typing not narrowed

    if survivors.empty:
        raise ValueError("No surviving models after degeneracy filters. Loosen thresholds or inspect why rejected_df is full.")

    # --- Soft-score bounds (from config, with defaults matching regime_config.yaml)
    _default_ss: Dict[str, Any] = {
        "duration":    {"optimal": 90.0, "lo": 20.0, "hi": 200.0, "slack": 0.75},
        "turnover":    {"optimal": 0.15, "lo": 0.05, "hi": 0.30,  "slack": 1.0},
        "persistence": {"optimal": 90.0, "lo": 20.0, "hi": 200.0, "slack": 1.0},
    }
    _cfg_ss: Dict[str, Any] = soft_score_cfg or {}
    ss: Dict[str, Any] = {
        k: {**_default_ss[k], **_cfg_ss.get(k, {})} for k in _default_ss
    }

    # --- Scoring
    # Percentile rank: maps a Series to [0,1] where higher rank = better model
    def rrank(s: pd.Series, ascending: bool = False) -> pd.Series:
        return s.rank(pct=True, ascending=ascending)

    # Macro coherence: higher Mahalanobis separation and R² are better
    macro = (
        0.5 * rrank(survivors["maha_median"], ascending=True)
        + 0.5 * rrank(survivors["anova_r2_mean"], ascending=True)
    )

    # Transition realism: prefer durations ~90 days (4-5 months), not too short/long
    # _soft_score replaces the flat-top _range_score so within-range models are ranked
    dur_score = survivors["median_implied_duration"].apply(  # type: ignore[arg-type]
        lambda x: _soft_score(x, **ss["duration"])
    )
    tv_score = (
        survivors["tv20"].apply(lambda x: _soft_score(x, **ss["turnover"]))  # type: ignore[arg-type]
        if survivors["tv20"].notna().any()
        else 0.0
    )
    off_pen = rrank(survivors["max_offdiag"], ascending=False)  # lower offdiag => higher rank
    trans = 0.45 * dur_score + 0.35 * tv_score + 0.20 * off_pen

    # Stability: less churn, decent persistence (~90 days), not overly certain
    churn = rrank(survivors["n_transitions"], ascending=False)
    pers = survivors["avg_persistence"].apply(  # type: ignore[arg-type]
        lambda x: _soft_score(x, **ss["persistence"])
    )
    ent = rrank(survivors["entropy_mean"], ascending=True)
    stab = 0.45 * churn + 0.35 * pers + 0.20 * ent

    # OOS robustness: macro coherence must not collapse out-of-sample
    oos_macro: pd.Series | float = (
        rrank(survivors["oos_anova_r2_mean"], ascending=True)
        if survivors["oos_anova_r2_mean"].notna().any()  # type: ignore[union-attr]  # pandas Series notna().any() — mypy cannot narrow after column indexing
        else 0.0
    )

    # BIC: lower is better — normalise within survivors, invert so higher = better
    bic_vals = survivors["bic_full"]
    if bic_vals.notna().any():  # type: ignore[union-attr]  # pandas Series notna().any() — mypy cannot narrow type
        bic_min, bic_max = float(bic_vals.min()), float(bic_vals.max())
        bic_score: pd.Series | float = (
            (bic_max - bic_vals) / (bic_max - bic_min)
            if bic_max > bic_min
            else pd.Series(1.0, index=survivors.index)
        )
    else:
        bic_score = 0.0

    survivors["macro_score"] = macro
    survivors["transition_score"] = trans
    survivors["stability_score"] = stab
    survivors["oos_macro_score"] = oos_macro
    survivors["bic_score"] = bic_score

    survivors["final_score"] = (
        w["macro"] * survivors["macro_score"]
        + w["transitions"] * survivors["transition_score"]
        + w["stability"] * survivors["stability_score"]
        + w["oos"] * survivors["oos_macro_score"]
        + w["bic"] * survivors["bic_score"]
    )

    leaderboard = survivors.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)  # type: ignore[assignment]  # pandas method chain returns DataFrame — mypy cannot narrow
    best_model_id = str(leaderboard.loc[0, "model_id"])

    return best_model_id, leaderboard, rejected_df