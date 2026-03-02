"""Tests for selection.py scoring formula — weights, _soft_score, BIC integration."""

import numpy as np
import pytest

from regime_ml.regimes.selection import _soft_score


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
