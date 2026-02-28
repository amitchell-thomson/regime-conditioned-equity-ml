"""Regime detection pipeline.

Assumes macro_features_ready.parquet already exists on disk (produced by
the feature pipeline). Runs the HMM grid search, model selection, regime
labeling, and episode validation, then writes all output artifacts.

Entry point: run_regime_pipeline()
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from regime_ml.utils.config import load_configs
from regime_ml.regimes.hmm import fit_best_of_n_seeds
from regime_ml.regimes.evaluation import (
    compare_hmm_models,
    expanding_window_cv,
    validate_against_episodes,
)
from regime_ml.regimes.selection import select_best_hmm_model
from regime_ml.regimes.labeling import label_regimes

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _resolve(path_str: str) -> Path:
    """Resolve a path string relative to the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _config_hash(cfg: dict) -> str:
    """SHA-256 of the YAML-serialisable regime config dict."""
    import yaml  # noqa: PLC0415

    raw = yaml.dump(cfg, default_flow_style=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _fit_grid(
    df_is: pd.DataFrame,
    grid_cfg: dict,
    hmm_cfg: dict,
    train_end_date: pd.Timestamp,
) -> Dict[str, Dict[str, Any]]:
    """Fit one HMMRegimeDetector per grid combination.

    Returns:
        models: {model_id: {"model": detector, "scaler": scaler,
                             "n_features": int, "split_date": str,
                             "n_regimes": int, "covariance_type": str, "p_stay": float}}
    """
    n_regimes_list: list[int] = grid_cfg.get("n_regimes", [3, 4])
    cov_types: list[str] = grid_cfg.get("covariance_types", ["diag"])
    p_stay_list: list[float] = grid_cfg.get("p_stay", [0.90, 0.95, 0.99])
    n_init: int = int(hmm_cfg.get("n_init", 10))
    min_regime_share: float = float(hmm_cfg.get("min_regime_share", 0.03))

    feature_names = list(df_is.columns)
    models: Dict[str, Dict[str, Any]] = {}

    for idx, (n_regimes, cov_type, p_stay) in enumerate(
        product(n_regimes_list, cov_types, p_stay_list)
    ):
        model_id = f"model_{idx}"
        logger.info(
            "_fit_grid: fitting %s (n_regimes=%d, cov=%s, p_stay=%.2f)",
            model_id,
            n_regimes,
            cov_type,
            p_stay,
        )
        try:
            detector, scaler = fit_best_of_n_seeds(
                df_is,
                n_regimes=n_regimes,
                n_init=n_init,
                train_end_date=train_end_date,
                covariance_type=cov_type,
                p_stay=p_stay,
                min_regime_share=min_regime_share,
                feature_names=feature_names,
            )
            models[model_id] = {
                "model": detector,
                "scaler": scaler,
                "n_features": len(feature_names),
                "split_date": str(train_end_date.date()),
                "n_regimes": n_regimes,
                "covariance_type": cov_type,
                "p_stay": p_stay,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_fit_grid: %s failed — %s: %s", model_id, type(exc).__name__, exc
            )

    if not models:
        raise RuntimeError(
            "_fit_grid: all grid configurations failed. Check features and HMM config."
        )
    logger.info(
        "_fit_grid: %d/%d configs fitted successfully.",
        len(models),
        len(n_regimes_list) * len(cov_types) * len(p_stay_list),
    )
    return models


def run_regime_pipeline(log_dir: Path | None = None) -> Dict[str, Any]:
    """Run the full regime detection pipeline.

    Reads macro_features_ready.parquet, fits an HMM grid, selects the best
    model, labels regimes, validates against known economic episodes, and
    writes all output artifacts to disk.

    Args:
        log_dir: Optional directory for a timestamped pipeline log file.
                 When supplied, ``configure_pipeline_logging(log_dir)`` is
                 called at the start so all regime_ml log output is persisted.

    Returns:
        Summary dict with keys: best_model_id, n_regimes, bic, final_score,
        n_matched_episodes, output_paths.
    """
    if log_dir is not None:
        from regime_ml.utils.logging import configure_pipeline_logging

        configure_pipeline_logging(log_dir)
    cfg = load_configs()
    regime_cfg: dict = cfg.get("regimes", {})

    train_end_date = pd.Timestamp(regime_cfg["train_end_date"])
    hmm_cfg: dict = regime_cfg.get("hmm", {})
    grid_cfg: dict = regime_cfg.get("hmm_grid", {})
    cv_cfg: dict = regime_cfg.get("cross_validation", {})
    outputs_cfg: dict = regime_cfg.get("outputs", {})
    sel_weights: dict | None = regime_cfg.get("selection", {}).get("weights")
    soft_score_cfg: dict | None = regime_cfg.get("selection", {}).get("soft_score")
    sel_filters: dict = regime_cfg.get("selection", {}).get("filters", {})

    # --- 1. Load features
    features_path = _resolve(
        outputs_cfg.get("features_path", "data/features/macro_features_ready.parquet")
    )
    logger.info("run_regime_pipeline: loading features from %s", features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}. Run 'regime-ml features' first."
        )
    features_df = pd.read_parquet(features_path)
    logger.info(
        "run_regime_pipeline: features shape %s, columns: %s",
        features_df.shape,
        list(features_df.columns),
    )

    # SHA-256 fingerprint of the feature data — changes if upstream FRED data is updated.
    features_hash = hashlib.sha256(
        pd.util.hash_pandas_object(features_df, index=True).values.tobytes()
    ).hexdigest()[:16]
    logger.info("run_regime_pipeline: features_hash=%s", features_hash)

    feature_names = list(features_df.columns)

    # --- 2. IS / OOS split
    df_is = features_df[features_df.index <= train_end_date]
    if len(df_is) < 100:
        raise ValueError(
            f"In-sample period contains only {len(df_is)} rows. "
            "Check train_end_date in regime_config.yaml."
        )
    logger.info(
        "run_regime_pipeline: IS=%d rows (%s → %s), OOS=%d rows",
        len(df_is),
        df_is.index[0].date(),
        df_is.index[-1].date(),
        len(features_df) - len(df_is),
    )

    # --- 3. Fit HMM grid
    models = _fit_grid(df_is, grid_cfg, hmm_cfg, train_end_date)

    # --- 4. Compare models
    results = compare_hmm_models(features_df, models)

    # --- 5. Model selection with optional CV-churn penalty
    # Build filter kwargs from YAML — only pass keys that select_best_hmm_model accepts
    _filter_kwarg_map = {
        "max_implied_duration": "max_implied_duration",
        "min_exit_paths": "min_exit_paths_required",
    }
    filter_kwargs = {
        kwarg: sel_filters[key]
        for key, kwarg in _filter_kwarg_map.items()
        if key in sel_filters
    }

    # 5a. Initial selection without churn to identify the strongest candidates.
    _initial_id, leaderboard_initial, rejected_df = select_best_hmm_model(
        results,
        weights=sel_weights,
        soft_score_cfg=soft_score_cfg,
        **filter_kwargs,
    )

    # 5b. Expanding-window CV for top-6 candidates (not all 12 grid models).
    # Top-6 ensures at least one model from each n_regimes group gets CV computed,
    # preventing the bias where untested models (neutral churn=0.5) score higher
    # than tested models that were found to be churny (real churn < 0.5 churn_stability).
    cv_results_by_model: Dict[str, Dict[str, Any]] = {}
    churn_scores: Dict[str, float] = {}
    if cv_cfg.get("enabled", False):
        top_candidates = list(leaderboard_initial["model_id"].head(6))
        logger.info(
            "run_regime_pipeline: computing expanding-window CV for top-%d candidates: %s",
            len(top_candidates),
            top_candidates,
        )
        for cand_id in top_candidates:
            cand_data = models[cand_id]
            cv_result = expanding_window_cv(
                features_df=features_df,
                model_config={
                    "n_regimes": cand_data["n_regimes"],
                    "covariance_type": cand_data["covariance_type"],
                    "p_stay": cand_data["p_stay"],
                },
                hmm_config=hmm_cfg,
                cv_config=cv_cfg,
                train_end_date=train_end_date,
            )
            cv_results_by_model[cand_id] = cv_result
            churn_scores[cand_id] = float(cv_result.get("label_churn") or 0.5)
            logger.info(
                "run_regime_pipeline: CV %s — label_churn=%.3f, n_folds=%d",
                cand_id,
                churn_scores[cand_id],
                cv_result.get("n_folds", 0),
            )

    # 5c. Final selection with churn penalty applied to top candidates.
    best_model_id, leaderboard_df, _ = select_best_hmm_model(
        results,
        weights=sel_weights,
        soft_score_cfg=soft_score_cfg,
        churn_scores=churn_scores or None,
        **filter_kwargs,
    )
    best_row = leaderboard_df.iloc[0]
    best_model_data = models[best_model_id]
    best_detector = best_model_data["model"]
    best_scaler = best_model_data["scaler"]
    logger.info(
        "run_regime_pipeline: best model = %s (n_regimes=%d, p_stay=%.2f, score=%.4f)",
        best_model_id,
        int(best_model_data["n_regimes"]),
        float(best_model_data["p_stay"]),
        float(best_row.get("final_score", np.nan)),
    )

    # --- 6. Regime assignments on full dataset
    X_full = best_scaler.transform(features_df.values)
    viterbi_labels = best_detector.predict(X_full)
    filter_proba_full = best_detector.filter_proba(X_full)

    regime_series = pd.Series(viterbi_labels, index=features_df.index, name="regime")

    # --- 7. Label regimes (uses smooth_proba for analysis quality)
    smooth_proba_full = best_detector.smooth_proba(X_full)
    label_results = label_regimes(X_full, smooth_proba_full, feature_names)
    state_to_label = {r["state_idx"]: r["label"] for r in label_results}

    # --- 8. Episode validation
    episodes_path = _PROJECT_ROOT / "configs/regimes/economic_episodes.yaml"
    try:
        episode_df = validate_against_episodes(
            regime_series, label_results, episodes_path
        )
        # Exclude episodes with no data overlap (e.g. pre-dataset history).
        # Counting them as failures inflates the miss rate against an unreachable target.
        reachable = (
            episode_df[episode_df["n_days"] > 0] if not episode_df.empty else episode_df
        )
        n_matched_episodes = (
            int(reachable["archetype_match"].sum()) if not reachable.empty else 0
        )
        n_episodes = len(reachable)
        logger.info(
            "run_regime_pipeline: episode validation — %d/%d matched.",
            n_matched_episodes,
            n_episodes,
        )
    except FileNotFoundError:
        logger.warning(
            "run_regime_pipeline: economic_episodes.yaml not found — skipping episode validation."
        )
        episode_df = pd.DataFrame()
        n_matched_episodes = 0
        n_episodes = 0

    # --- 10. Build output DataFrames / dicts
    # Regime assignments parquet
    K = best_detector.n_regimes
    regime_label_series = regime_series.map(state_to_label).rename("regime_label")
    proba_cols = {f"filter_proba_{k}": filter_proba_full[:, k] for k in range(K)}
    regime_assignments_df = pd.concat(
        [
            regime_series,
            regime_label_series,
            pd.DataFrame(proba_cols, index=features_df.index),
        ],
        axis=1,
    )

    # Run metadata
    run_metadata: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": _config_hash(regime_cfg),
        "features_hash": features_hash,
        "best_model_id": best_model_id,
        "n_regimes": int(best_model_data["n_regimes"]),
        "covariance_type": str(best_model_data["covariance_type"]),
        "p_stay": float(best_model_data["p_stay"]),
        "bic_is": (
            float(best_row.get("bic_is", np.nan))
            if np.isfinite(float(best_row.get("bic_is", np.nan) or float("nan")))
            else None
        ),
        "final_score": float(best_row.get("final_score", np.nan)),
        "train_end_date": str(train_end_date.date()),
        "data_start": str(features_df.index[0].date()),
        "data_end": str(features_df.index[-1].date()),
        "n_obs_is": len(df_is),
        "n_obs_total": len(features_df),
        "label_results": label_results,
        "episode_validation": {
            "n_episodes": n_episodes,
            "n_matched": n_matched_episodes,
        },
        "cv_diagnostics": cv_results_by_model.get(best_model_id),
    }

    # Log regime counts per label
    for r in label_results:
        count = int((regime_series == r["state_idx"]).sum())
        logger.info(
            "  State %d: %s — %d days (%.1f%%)",
            r["state_idx"],
            r["label"],
            count,
            100.0 * count / len(regime_series),
        )

    # --- 11. Save outputs
    out_dir = _resolve(
        outputs_cfg.get("regime_assignments", "data/regimes/regime_assignments.parquet")
    ).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = _resolve(
        outputs_cfg.get("regime_assignments", "data/regimes/regime_assignments.parquet")
    )
    leaderboard_path = _resolve(
        outputs_cfg.get("model_leaderboard", "data/regimes/model_leaderboard.csv")
    )
    label_path = _resolve(
        outputs_cfg.get("label_results", "data/regimes/label_results.json")
    )
    metadata_path = _resolve(
        outputs_cfg.get("run_metadata", "data/regimes/run_metadata.json")
    )

    # Persist the best model parameters so Phase 3 can load without re-fitting.
    best_model_path = _resolve(outputs_cfg.get("best_model", "data/regimes/best_model"))
    best_detector.save(best_model_path)
    logger.info("run_regime_pipeline: saved best model → %s", best_model_path)

    regime_assignments_df.to_parquet(assignments_path)
    logger.info("run_regime_pipeline: saved regime assignments → %s", assignments_path)

    leaderboard_df.to_csv(leaderboard_path, index=False)
    if not rejected_df.empty:
        rejected_path = leaderboard_path.parent / "model_rejected.csv"
        rejected_df.to_csv(rejected_path, index=False)
    logger.info("run_regime_pipeline: saved leaderboard → %s", leaderboard_path)

    with open(label_path, "w") as fh:
        json.dump(label_results, fh, indent=2, default=str)
    logger.info("run_regime_pipeline: saved label results → %s", label_path)

    with open(metadata_path, "w") as fh:
        json.dump(run_metadata, fh, indent=2, default=str)
    logger.info("run_regime_pipeline: saved run metadata → %s", metadata_path)

    return {
        "best_model_id": best_model_id,
        "n_regimes": int(best_model_data["n_regimes"]),
        "bic_is": run_metadata["bic_is"],
        "final_score": run_metadata["final_score"],
        "n_matched_episodes": n_matched_episodes,
        "output_paths": {
            "regime_assignments": str(assignments_path),
            "model_leaderboard": str(leaderboard_path),
            "label_results": str(label_path),
            "run_metadata": str(metadata_path),
        },
    }
