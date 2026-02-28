"""Tests for fit_best_of_n_seeds() (item 3.1)."""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from regime_ml.regimes.hmm import (
    HMMRegimeDetector,
    fit_best_of_n_seeds,
    initialise_emissions,
    initialise_probabilities,
    initialise_transitions,
)


def _make_df(n: int = 300, d: int = 3, seed: int = 0) -> pd.DataFrame:
    """Synthetic feature DataFrame with a business-day DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-01", periods=n)
    cols = [f"f{i}" for i in range(d)]
    return pd.DataFrame(rng.standard_normal((n, d)), index=idx, columns=cols)


def test_returns_fitted_detector_and_scaler():
    """fit_best_of_n_seeds must return an (HMMRegimeDetector, StandardScaler) tuple."""
    df = _make_df()
    detector, scaler = fit_best_of_n_seeds(df, n_regimes=2, n_init=3)
    assert isinstance(detector, HMMRegimeDetector)
    assert isinstance(scaler, StandardScaler)
    assert detector.is_fitted


def test_deterministic_for_fixed_n_init():
    """Two calls with identical arguments must produce identical means."""
    df = _make_df(n=300, seed=0)
    det1, _ = fit_best_of_n_seeds(df, n_regimes=2, n_init=4)
    det2, _ = fit_best_of_n_seeds(df, n_regimes=2, n_init=4)
    np.testing.assert_allclose(
        det1.model.means_,
        det2.model.means_,
        atol=1e-8,
        err_msg="fit_best_of_n_seeds must be deterministic for the same n_init.",
    )


def test_best_of_n_ll_is_optimal():
    """The returned model must have the highest LL among seeds that pass the
    transmat validity filter. Seeds with absorbing states (tv_distance_valid=False)
    are correctly skipped even when they attain higher LL — the non-stationary
    transition matrix makes their labels economically meaningless."""
    from regime_ml.regimes.evaluation import evaluate_transmat_sanity

    df = _make_df(n=400, seed=7)
    n_seeds = 5
    detector, scaler = fit_best_of_n_seeds(
        df, n_regimes=2, n_init=n_seeds, min_regime_share=0.0
    )
    X_scaled = scaler.transform(df.values)
    best_ll = detector.score(X_scaled)

    # Re-run all seeds individually. Only compare against seeds that also have a
    # valid stationary distribution — those are the ones fit_best_of_n_seeds considers.
    for seed in range(n_seeds):
        means, covs, sc = initialise_emissions(df, n_clusters=2, random_state=seed)
        det = HMMRegimeDetector(
            n_regimes=2,
            random_state=seed,
            means=means,
            covars=covs,
            transmat=initialise_transitions(2, p_stay=0.9),
            startprob=initialise_probabilities(2),
        )
        det.fit(sc.transform(df.values))
        seed_ll = det.score(sc.transform(df.values))
        sanity = evaluate_transmat_sanity(det.model.transmat_, n_mix=1)
        if not sanity["tv_distance_valid"]:
            # Correctly excluded: absorbing state; LL comparison not meaningful.
            continue
        assert best_ll >= seed_ll - 1e-6, (
            f"Returned LL {best_ll:.4f} is below valid-transmat seed {seed} LL {seed_ll:.4f}."
        )


def test_degeneracy_filter_fallback_warning(caplog):
    """When all seeds fail the filter, a WARNING is logged and a model is still returned."""
    df = _make_df(n=200, seed=1)
    # min_regime_share=0.99 is impossible for n_regimes=2 → all seeds fail filter.
    with caplog.at_level(logging.WARNING, logger="regime_ml"):
        detector, _ = fit_best_of_n_seeds(
            df, n_regimes=2, n_init=3, min_regime_share=0.99
        )
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("degeneracy filter" in m for m in warning_messages)
    assert detector.is_fitted


def test_n_init_seeds_all_logged(caplog):
    """Each seed attempt must produce an INFO log entry containing 'seed=' and 'll='."""
    df = _make_df(n=200, seed=2)
    with caplog.at_level(logging.INFO, logger="regime_ml"):
        fit_best_of_n_seeds(df, n_regimes=2, n_init=3)
    seed_logs = [
        r
        for r in caplog.records
        if "fit_best_of_n_seeds" in r.message
        and "seed=" in r.message
        and "ll=" in r.message
    ]
    assert len(seed_logs) >= 3, (
        f"Expected at least 3 per-seed log entries, got {len(seed_logs)}."
    )


def test_is_boundary_enforced():
    """train_end_date inside the data range must propagate and raise ValueError."""
    df = _make_df(n=100, seed=3)
    # Set train_end_date to the 30th row — data extends beyond it.
    train_end = df.index[29]
    with pytest.raises(ValueError, match="train_end_date"):
        fit_best_of_n_seeds(df, n_regimes=2, n_init=2, train_end_date=train_end)


def test_feature_names_stored_on_detector():
    """feature_names passed to fit_best_of_n_seeds must be on the returned detector."""
    df = _make_df(d=3)
    names = ["alpha", "beta", "gamma"]
    detector, _ = fit_best_of_n_seeds(df, n_regimes=2, n_init=2, feature_names=names)
    assert detector.feature_names == names


def test_returns_correct_scaler():
    """The returned scaler must be able to transform the training data without error."""
    df = _make_df()
    _, scaler = fit_best_of_n_seeds(df, n_regimes=2, n_init=3)
    transformed = scaler.transform(df.values)
    assert transformed.shape == df.shape
    assert np.isfinite(transformed).all()
