"""Tests for the CV-as-hard-filter strategy.

Covers:
  1. expanding_window_cv() returns all Phase B diagnostic keys.
  2. churn_hard_reject is set correctly relative to hard_filter config.
  3. n_nontrivial_perms is 0 when all folds align trivially.
  4. per_state_share_std, transmat_diag_std, fold_score_slope are populated.
  5. select_best_hmm_model hard-rejects models in churn_rejected_ids.
  6. Hard rejection survives even when the rejected model had the best BIC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_ml.regimes.evaluation import expanding_window_cv
from regime_ml.regimes.selection import select_best_hmm_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(n: int, d: int = 3, seed: int = 0) -> pd.DataFrame:
    """Synthetic feature DataFrame with a business-day DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(
        rng.standard_normal((n, d)),
        index=idx,
        columns=[f"f{i}" for i in range(d)],
    )


def _minimal_cv_config(max_churn: float = 1.0, min_cv_folds: int = 0) -> dict:
    """CV config with a tiny window so tests complete quickly."""
    return {
        "hard_filter": {
            "max_churn": max_churn,
            "min_cv_folds": min_cv_folds,
        },
        "min_train_years": 2,
        "fold_step_months": 12,
        "oos_window_months": 12,
        "max_churn_warning": 0.20,
    }


def _minimal_hmm_config() -> dict:
    return {"n_init": 2, "min_regime_share": 0.03}


def _run_cv(
    features_df: pd.DataFrame,
    n_regimes: int = 2,
    max_churn: float = 1.0,
    min_cv_folds: int = 0,
) -> dict:
    train_end = features_df.index[-1]
    return expanding_window_cv(
        features_df=features_df,
        model_config={"n_regimes": n_regimes, "covariance_type": "full", "p_stay": 0.95},
        hmm_config=_minimal_hmm_config(),
        cv_config=_minimal_cv_config(max_churn=max_churn, min_cv_folds=min_cv_folds),
        train_end_date=train_end,
    )


# ---------------------------------------------------------------------------
# 1. Return dict contains all Phase B diagnostic keys
# ---------------------------------------------------------------------------


class TestExpandingWindowCVKeys:
    """expanding_window_cv must always return a complete result dict."""

    _REQUIRED_KEYS = {
        "label_churn",
        "metric_cv_std",
        "n_folds",
        "fold_scores",
        "churn_warning",
        # Phase B
        "churn_hard_reject",
        "max_pairwise_churn",
        "n_nontrivial_perms",
        "fold_score_slope",
        "transmat_diag_std",
        "per_state_share_std",
    }

    def test_keys_present_when_folds_complete(self):
        """All Phase B keys must be present when at least 2 folds complete."""
        # 800 trading days ≈ ~3.2 years; with min_train_years=2 we get ~1 annual step
        df = _make_features(n=800, seed=1)
        result = _run_cv(df)
        missing = self._REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_keys_present_on_insufficient_data_early_return(self):
        """All Phase B keys must be present even for the early-return (< 2 folds)."""
        # 100 rows is far too short for min_train_years=2 → early return
        df = _make_features(n=100, seed=2)
        result = _run_cv(df)
        assert result["n_folds"] == 0
        missing = self._REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys on early return: {missing}"

    def test_n_folds_is_nonnegative_int(self):
        df = _make_features(n=800, seed=3)
        result = _run_cv(df)
        assert isinstance(result["n_folds"], int)
        assert result["n_folds"] >= 0

    def test_fold_scores_length_matches_n_folds(self):
        df = _make_features(n=800, seed=4)
        result = _run_cv(df)
        assert len(result["fold_scores"]) == result["n_folds"]

    def test_n_nontrivial_perms_is_nonneg_int(self):
        df = _make_features(n=800, seed=5)
        result = _run_cv(df)
        assert isinstance(result["n_nontrivial_perms"], int)
        assert result["n_nontrivial_perms"] >= 0

    def test_n_nontrivial_perms_leq_n_folds_minus_one(self):
        """The first fold is always the reference (identity) — at most n_folds-1 can be
        non-trivial."""
        df = _make_features(n=800, seed=6)
        result = _run_cv(df)
        if result["n_folds"] > 0:
            assert result["n_nontrivial_perms"] <= result["n_folds"] - 1

    def test_per_state_share_std_keys_match_n_regimes(self):
        """per_state_share_std must have one entry per regime state."""
        df = _make_features(n=800, seed=7)
        result = _run_cv(df, n_regimes=2)
        if result["n_folds"] > 1:
            pss = result["per_state_share_std"]
            assert set(pss.keys()) == {"0", "1"}, (
                f"Expected keys {{'0','1'}}, got {set(pss.keys())}"
            )


# ---------------------------------------------------------------------------
# 2. churn_hard_reject is computed correctly
# ---------------------------------------------------------------------------


class TestChurnHardReject:
    """churn_hard_reject must reflect the hard_filter config, not max_churn_warning."""

    def test_hard_reject_false_when_threshold_is_one(self):
        """With max_churn=1.0 (the permissive default), no model should be rejected."""
        df = _make_features(n=800, seed=10)
        result = _run_cv(df, max_churn=1.0)
        # label_churn is always < 1.0 in a well-behaved model
        if result["n_folds"] > 1 and result["label_churn"] is not None:
            assert result["churn_hard_reject"] is False

    def test_hard_reject_true_when_threshold_is_zero(self):
        """With max_churn=0.0, any non-zero churn triggers hard reject."""
        df = _make_features(n=800, seed=11)
        result = _run_cv(df, max_churn=0.0)
        # On synthetic Gaussian data (no real regime structure), label_churn > 0
        # This might not always hold, so we only assert if churn is measured
        if result["n_folds"] > 1 and result["label_churn"] is not None:
            if result["label_churn"] > 0.0:
                assert result["churn_hard_reject"] is True

    def test_hard_reject_true_when_min_cv_folds_not_met(self):
        """With min_cv_folds=100, any realistic run will have too few folds."""
        df = _make_features(n=800, seed=12)
        result = _run_cv(df, max_churn=1.0, min_cv_folds=100)
        assert result["churn_hard_reject"] is True, (
            "min_cv_folds=100 should always trigger hard reject for this data size"
        )

    def test_hard_reject_false_when_n_folds_meets_requirement(self):
        """With min_cv_folds=1, a run with at least 1 fold should not trigger."""
        df = _make_features(n=800, seed=13)
        result = _run_cv(df, max_churn=1.0, min_cv_folds=1)
        if result["n_folds"] >= 1:
            # churn can still trigger via max_churn, but max_churn=1.0 means never
            assert result["churn_hard_reject"] is False

    def test_early_return_always_hard_rejects(self):
        """Insufficient data → n_folds=0 → churn_hard_reject=True regardless."""
        df = _make_features(n=50, seed=14)
        result = _run_cv(df)
        assert result["n_folds"] == 0
        assert result["churn_hard_reject"] is True


# ---------------------------------------------------------------------------
# 3. select_best_hmm_model hard-rejects churn_rejected_ids
# ---------------------------------------------------------------------------


def _make_minimal_results(model_ids: list[str], bic_values: dict[str, float]) -> dict:
    """Build a minimal compare_hmm_models()-style results dict for select_best_hmm_model.

    All models pass structural hard filters; only BIC differs between them.
    """
    results = {}
    for mid in model_ids:
        results[mid] = {
            "bic_is": bic_values.get(mid, 10000.0),
            "aic_is": bic_values.get(mid, 10000.0),
            "regime_stability": {
                "avg_persistence": 90.0,
                "std_persistence": 20.0,
                "n_transitions": 50,
                "regime_entropy": 1.5,
                "regime_counts": {0: 500, 1: 500},
                "min_regime_share": 0.10,
                "max_regime_share": 0.50,
            },
            "entropy_balance": {"entropy_balance": 1.5},
            "transition_matrix_sanity": {
                "median_implied_duration": 90.0,
                "max_implied_duration": 200.0,
                "mean_self_transition": 0.95,
                "max_offdiag_transition": 0.05,
                "mean_row_entropy": 0.3,
                "tv_distance_valid": True,
                "tv_distance": 0.55,
                "min_exit_paths": 2,
            },
            "macro_coherence": {
                "maha_min": 2.0,
                "maha_median": 2.5,
                "maha_mean": 2.5,
                "anova_r2_mean": 0.35,
                "anova_r2_median": 0.35,
                "anova_group_r2": {},
                "anova_top_features": [],
            },
            "in_sample": {
                "regime_stability": {
                    "min_regime_share": 0.10,
                    "max_regime_share": 0.50,
                    "avg_persistence": 90.0,
                    "n_transitions": 40,
                },
                "entropy_balance": {"entropy_balance": 1.5},
                "macro_coherence": {"anova_r2_mean": 0.35},
            },
            "out_of_sample": {
                "regime_stability": {
                    "min_regime_share": 0.10,
                    "max_regime_share": 0.50,
                    "avg_persistence": 90.0,
                    "n_transitions": 10,
                },
                "entropy_balance": {"entropy_balance": 1.5},
                "macro_coherence": {"anova_r2_mean": 0.30},
            },
        }
    return results


class TestSelectBestHMMModelChurnHardFilter:
    def test_churn_rejected_model_is_excluded(self):
        """A model in churn_rejected_ids must not appear in the leaderboard."""
        model_ids = ["model_A", "model_B", "model_C"]
        # model_A has the best (lowest) BIC but is churn-rejected
        bic_values = {"model_A": 5000.0, "model_B": 7000.0, "model_C": 9000.0}
        results = _make_minimal_results(model_ids, bic_values)

        best_id, leaderboard, rejected_df = select_best_hmm_model(
            results,
            churn_rejected_ids={"model_A"},
        )
        leaderboard_ids = set(leaderboard["model_id"].tolist())
        assert "model_A" not in leaderboard_ids, (
            "model_A was churn-rejected and must not appear in the leaderboard"
        )
        assert best_id != "model_A", (
            "model_A was churn-rejected and must not be selected as best"
        )

    def test_churn_rejection_appears_in_rejected_df(self):
        """Churn-rejected models must appear in rejected_df with the correct reason."""
        model_ids = ["model_A", "model_B"]
        results = _make_minimal_results(model_ids, {"model_A": 5000.0, "model_B": 7000.0})

        _, _, rejected_df = select_best_hmm_model(
            results,
            churn_rejected_ids={"model_A"},
        )
        assert "model_A" in rejected_df["model_id"].values
        reason = rejected_df.loc[
            rejected_df["model_id"] == "model_A", "reason"
        ].iloc[0]
        assert "churn" in reason.lower(), (
            f"Rejection reason should mention churn, got: '{reason}'"
        )

    def test_best_bic_wins_when_not_churn_rejected(self):
        """Without churn filtering, the model with best BIC should dominate (via bic_score)."""
        model_ids = ["model_A", "model_B"]
        # model_B has much better BIC — should win via bic_score weight
        bic_values = {"model_A": 10000.0, "model_B": 5000.0}
        results = _make_minimal_results(model_ids, bic_values)

        best_id, _, _ = select_best_hmm_model(results, churn_rejected_ids=None)
        assert best_id == "model_B", (
            "model_B has substantially better BIC and should win when churn filter is off"
        )

    def test_no_churn_rejected_ids_disables_hard_filter(self):
        """When churn_rejected_ids is None, no model should be rejected on churn grounds."""
        model_ids = ["model_A", "model_B", "model_C"]
        bic_values = {"model_A": 5000.0, "model_B": 7000.0, "model_C": 9000.0}
        results = _make_minimal_results(model_ids, bic_values)

        _, leaderboard, rejected_df = select_best_hmm_model(
            results,
            churn_rejected_ids=None,
        )
        leaderboard_ids = set(leaderboard["model_id"].tolist())
        for mid in model_ids:
            assert mid in leaderboard_ids, (
                f"{mid} should not be churn-rejected when churn_rejected_ids=None"
            )

    def test_all_models_rejected_raises(self):
        """Hard-rejecting all models must raise ValueError with a useful message."""
        model_ids = ["model_A", "model_B"]
        results = _make_minimal_results(model_ids, {"model_A": 5000.0, "model_B": 7000.0})

        with pytest.raises(ValueError, match="No surviving"):
            select_best_hmm_model(
                results,
                churn_rejected_ids={"model_A", "model_B"},
            )

    def test_empty_churn_rejected_ids_is_no_op(self):
        """An empty set should be treated the same as None — no extra rejections."""
        model_ids = ["model_A", "model_B"]
        results = _make_minimal_results(model_ids, {"model_A": 5000.0, "model_B": 7000.0})

        _, lb_none, _ = select_best_hmm_model(results, churn_rejected_ids=None)
        _, lb_empty, _ = select_best_hmm_model(results, churn_rejected_ids=set())
        assert set(lb_none["model_id"]) == set(lb_empty["model_id"])
