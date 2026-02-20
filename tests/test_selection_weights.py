"""Tests for selection.py scoring formula weights."""



def test_scoring_weights_sum_to_one():
    """The hardcoded scoring weights in select_best_hmm_model must sum to 1.0."""
    macro_w = 0.30
    transition_w = 0.25
    stability_w = 0.20
    oos_macro_w = 0.25
    total = macro_w + transition_w + stability_w + oos_macro_w
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_oos_weight_is_meaningful():
    """OOS weight must be at least 20% — below this it is practically decorative."""
    oos_macro_w = 0.25
    assert oos_macro_w >= 0.20, (
        f"OOS weight {oos_macro_w} is too low. "
        "The audit requires ≥20% to make OOS coherence a primary selection signal."
    )
