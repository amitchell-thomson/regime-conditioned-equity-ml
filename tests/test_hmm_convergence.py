"""Tests for HMM convergence diagnostics, Cholesky jitter, and random_state."""

import logging
import numpy as np
import pandas as pd

from regime_ml.regimes.hmm import HMMRegimeDetector, initialise_emissions


def _make_synthetic_data(n_samples: int = 100, n_features: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def test_convergence_warning_fires(caplog):
    """A deliberately under-iterated HMM should log a WARNING about non-convergence."""
    X = _make_synthetic_data(n_samples=80, n_features=2)
    # tol=1e-300 is impossible to satisfy; n_iter=3 gives >=2 history entries so
    # hmmlearn's convergence check can actually evaluate the LL delta vs tol.
    detector = HMMRegimeDetector(n_regimes=3, n_iter=3, tol=1e-300, random_state=42)

    with caplog.at_level(logging.WARNING, logger="regime_ml"):
        detector.fit(X)

    assert any(
        "did not converge" in r.message for r in caplog.records
    ), "Expected a convergence WARNING log record from regime_ml.regimes.hmm"


def test_convergence_warning_not_fired_when_converged(caplog):
    """A well-configured HMM that converges should not emit a convergence warning."""
    X = _make_synthetic_data(n_samples=200, n_features=2)
    detector = HMMRegimeDetector(n_regimes=2, n_iter=200, tol=1e-3, random_state=42)

    with caplog.at_level(logging.WARNING, logger="regime_ml"):
        detector.fit(X)

    convergence_warnings = [r for r in caplog.records if "did not converge" in r.message]
    assert len(convergence_warnings) == 0


def test_random_state_none_accepted():
    """HMMRegimeDetector should accept random_state=None without error."""
    X = _make_synthetic_data(n_samples=100, n_features=2)
    detector = HMMRegimeDetector(n_regimes=2, n_iter=50, random_state=None)
    detector.fit(X)
    assert detector.is_fitted


def test_initialise_emissions_random_state_none():
    """initialise_emissions should accept random_state=None."""
    df_train = pd.DataFrame(
        np.random.default_rng(1).standard_normal((80, 3)),
        columns=["f1", "f2", "f3"],
    )
    means, covs, scaler = initialise_emissions(df_train, n_clusters=2, random_state=None)
    assert means.shape == (2, 3)
    assert covs.shape == (2, 3, 3)


def test_cholesky_jitter_produces_pd_covariances():
    """Covariances from initialise_emissions should be positive-definite (Cholesky decomposable)."""
    rng = np.random.default_rng(7)
    # Create data where one cluster has very few points (near-singular covariance risk)
    X = np.vstack(
        [
            rng.standard_normal((5, 4)),  # tiny cluster
            rng.standard_normal((95, 4)) + 5.0,
        ]
    )
    df_train = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    _, covs, _ = initialise_emissions(df_train, n_clusters=2, random_state=42)

    for k in range(covs.shape[0]):
        # np.linalg.cholesky raises LinAlgError if not positive definite
        np.linalg.cholesky(covs[k])
