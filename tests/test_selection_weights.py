"""Tests for selection.py scoring formula — weights, _soft_score, BIC integration."""

import numpy as np
import pytest

from regime_ml.regimes.selection import _soft_score, select_best_hmm_model


# ---------------------------------------------------------------------------
# Shared fixture factory for transition-score tests
# ---------------------------------------------------------------------------


def _make_result_with_tv20(tv20: float, bic: float = 8000.0) -> dict:
    """Minimal compare_hmm_models()-style result dict, tv20 controllable."""
    return {
        "bic_is": bic,
        "aic_is": bic,
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
            "tv_distance": tv20,
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


# ---------------------------------------------------------------------------
# Weight integrity
# ---------------------------------------------------------------------------


class TestScoringWeights:
    def test_default_weights_sum_to_one(self):
        """The five default scoring weights in select_best_hmm_model must sum to 1.0.

        CV churn is now a hard filter (churn_rejected_ids), not a soft score.
        Its former 0.15 weight is redistributed: macro +0.05, oos +0.05, bic +0.05.
        """
        macro_w = 0.25
        transitions_w = 0.20
        stability_w = 0.20
        oos_w = 0.20
        bic_w = 0.15
        total = macro_w + transitions_w + stability_w + oos_w + bic_w
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_bic_weight_nonzero(self):
        """BIC weight must be > 0 — it was added to penalise model complexity."""
        bic_w = 0.15
        assert bic_w > 0.0, "BIC weight must be strictly positive."

    def test_bic_weight_meaningful(self):
        """BIC weight must be at least 0.10 — it needs to overcome macro score gaps."""
        bic_w = 0.15
        assert bic_w >= 0.10, (
            f"BIC weight {bic_w} is too low to discriminate between models with "
            "different n_regimes when BIC differences are large (>1000)."
        )

    def test_no_churn_weight(self):
        """CV churn must not appear in scoring weights — it is now a hard filter."""
        from regime_ml.regimes.selection import select_best_hmm_model
        import inspect

        sig = inspect.signature(select_best_hmm_model)
        # churn_scores param must not exist; churn_rejected_ids must exist
        assert (
            "churn_scores" not in sig.parameters
        ), "churn_scores parameter should have been replaced by churn_rejected_ids"
        assert (
            "churn_rejected_ids" in sig.parameters
        ), "churn_rejected_ids parameter must be present for hard churn filtering"


# ---------------------------------------------------------------------------
# _soft_score tests
# ---------------------------------------------------------------------------


class TestSoftScore:
    def test_returns_one_at_optimal(self):
        assert _soft_score(90.0, optimal=90.0, lo=20.0, hi=200.0) == pytest.approx(1.0)

    def test_returns_point_five_at_lo_boundary(self):
        """lo boundary should return 0.5 (start of within-range gradient)."""
        score = _soft_score(20.0, optimal=90.0, lo=20.0, hi=200.0)
        assert score == pytest.approx(0.5)

    def test_returns_point_five_at_hi_boundary(self):
        score = _soft_score(200.0, optimal=90.0, lo=20.0, hi=200.0)
        assert score == pytest.approx(0.5)

    def test_score_between_lo_and_optimal_is_in_range(self):
        score = _soft_score(55.0, optimal=90.0, lo=20.0, hi=200.0)
        assert 0.5 <= score <= 1.0

    def test_score_between_optimal_and_hi_is_in_range(self):
        score = _soft_score(145.0, optimal=90.0, lo=20.0, hi=200.0)
        assert 0.5 <= score <= 1.0

    def test_score_below_lo_minus_slack_is_zero(self):
        """Well outside the range should hit zero."""
        # slack=0.5, width=180, so zero at lo - 0.5*180 = 20 - 90 = -70
        score = _soft_score(-200.0, optimal=90.0, lo=20.0, hi=200.0, slack=0.5)
        assert score == 0.0

    def test_score_above_hi_plus_slack_is_zero(self):
        score = _soft_score(500.0, optimal=90.0, lo=20.0, hi=200.0, slack=0.5)
        assert score == 0.0

    def test_score_is_monotone_increasing_toward_optimal_from_lo(self):
        """As v increases from lo toward optimal, score must increase."""
        lo, opt, hi = 20.0, 90.0, 200.0
        xs = np.linspace(lo, opt, 20)
        scores = [_soft_score(x, optimal=opt, lo=lo, hi=hi) for x in xs]
        diffs = np.diff(scores)
        assert np.all(diffs >= -1e-9), f"Not monotone increasing: {scores}"

    def test_score_is_monotone_decreasing_from_optimal_to_hi(self):
        lo, opt, hi = 20.0, 90.0, 200.0
        xs = np.linspace(opt, hi, 20)
        scores = [_soft_score(x, optimal=opt, lo=lo, hi=hi) for x in xs]
        diffs = np.diff(scores)
        assert np.all(diffs <= 1e-9), f"Not monotone decreasing: {scores}"

    def test_non_finite_returns_zero(self):
        assert _soft_score(float("nan"), optimal=90.0, lo=20.0, hi=200.0) == 0.0
        assert _soft_score(float("inf"), optimal=90.0, lo=20.0, hi=200.0) == 0.0

    def test_zero_width_returns_zero(self):
        assert _soft_score(50.0, optimal=50.0, lo=50.0, hi=50.0) == 0.0

    def test_models_within_range_are_differentiated(self):
        """Two models both inside [lo, hi] must receive different scores when
        one is closer to optimal — this was the bug in the old flat-top _range_score."""
        score_closer = _soft_score(85.0, optimal=90.0, lo=20.0, hi=200.0)
        score_farther = _soft_score(30.0, optimal=90.0, lo=20.0, hi=200.0)
        assert score_closer > score_farther, (
            "Model closer to optimal should score higher than one farther from it "
            "even when both are inside [lo, hi]."
        )


# ---------------------------------------------------------------------------
# TV-20 removal regression: transition_score must be independent of tv20
# ---------------------------------------------------------------------------


class TestTransitionScoreNoTV:
    """tv_score was removed from the transition formula because the TV-20 mixing
    distance at p_stay in [0.93, 0.99] is 0.47-0.65 — well above the former
    hi=0.30 threshold, producing score=0.0 for every model in the grid.

    After the fix, transition_score = 0.65 * dur_score + 0.35 * off_pen.
    Changing tv20 while keeping all other model properties fixed must NOT
    change the transition_score.
    """

    def _run(self, tv20: float) -> float:
        results = {
            "model_A": _make_result_with_tv20(tv20, bic=8000.0),
            "model_B": _make_result_with_tv20(tv20, bic=8001.0),
        }
        _, lb, _ = select_best_hmm_model(results, churn_rejected_ids=None)
        return float(lb.loc[lb["model_id"] == "model_A", "transition_score"].iloc[0])

    def test_low_tv20_same_transition_score_as_high_tv20(self):
        """A tv20 of 0.10 (formerly inside thresholds) and 0.65 (outside) must
        both yield the same transition_score — tv_score no longer contributes."""
        score_low = self._run(tv20=0.10)
        score_high = self._run(tv20=0.65)
        assert score_low == pytest.approx(score_high, abs=1e-9), (
            f"transition_score differs between tv20=0.10 ({score_low:.6f}) and "
            f"tv20=0.65 ({score_high:.6f}). tv20 must not affect transition_score."
        )

    def test_extreme_tv20_does_not_change_transition_score(self):
        """tv20=0.0 and tv20=0.99 should produce identical transition_scores."""
        score_zero = self._run(tv20=0.0)
        score_max = self._run(tv20=0.99)
        assert score_zero == pytest.approx(score_max, abs=1e-9), (
            "Extreme tv20 values should not affect transition_score."
        )

    def test_transition_score_column_present_in_leaderboard(self):
        """transition_score column must still exist in the leaderboard output."""
        results = {
            "model_A": _make_result_with_tv20(0.55, bic=8000.0),
            "model_B": _make_result_with_tv20(0.60, bic=8001.0),
        }
        _, lb, _ = select_best_hmm_model(results, churn_rejected_ids=None)
        assert "transition_score" in lb.columns, (
            "transition_score column must be present in leaderboard."
        )

    def test_transition_score_is_positive(self):
        """transition_score must be > 0 for a model with typical dur and offdiag values."""
        results = {
            "model_A": _make_result_with_tv20(0.55, bic=8000.0),
            "model_B": _make_result_with_tv20(0.55, bic=8001.0),
        }
        _, lb, _ = select_best_hmm_model(results, churn_rejected_ids=None)
        ts = float(lb.loc[lb["model_id"] == "model_A", "transition_score"].iloc[0])
        assert ts > 0.0, (
            f"transition_score should be positive for a well-behaved model; got {ts}"
        )
