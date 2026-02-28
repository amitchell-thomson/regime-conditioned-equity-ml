"""Tests for Ledoit-Wolf covariance regularization in initialise_emissions (item 3.2)."""

import logging

import numpy as np
import pandas as pd

from regime_ml.regimes.hmm import initialise_emissions


def _make_df(n: int, d: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(d)]
    return pd.DataFrame(rng.standard_normal((n, d)), columns=cols)


def test_lw_covariances_are_positive_definite():
    """All returned covariances must be Cholesky-decomposable (strictly PD)."""
    df = _make_df(200, d=4)
    _, covs, _ = initialise_emissions(df, n_clusters=3, random_state=42)
    for k in range(covs.shape[0]):
        # Raises np.linalg.LinAlgError if matrix is not PD.
        np.linalg.cholesky(covs[k])


def test_lw_covariances_are_symmetric():
    """All returned covariances must be symmetric."""
    df = _make_df(150, d=3)
    _, covs, _ = initialise_emissions(df, n_clusters=3, random_state=42)
    for k in range(covs.shape[0]):
        np.testing.assert_allclose(covs[k], covs[k].T, atol=1e-10)


def test_lw_covariances_finite():
    """All covariance elements must be finite."""
    df = _make_df(100, d=4)
    _, covs, _ = initialise_emissions(df, n_clusters=3, random_state=42)
    assert np.isfinite(covs).all()


def test_rank_deficient_cluster_is_still_pd():
    """Ledoit-Wolf handles n_points < n_features (rank-deficient case) and returns PD cov."""
    # 3 features, cluster with only 2 points — rank-deficient sample covariance
    rng = np.random.default_rng(7)
    X = np.vstack(
        [
            rng.standard_normal((2, 3)),  # tiny cluster — rank-deficient sample cov
            rng.standard_normal((200, 3)) + 10.0,  # large, well-separated cluster
        ]
    )
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    _, covs, _ = initialise_emissions(df, n_clusters=2, random_state=0)
    for k in range(covs.shape[0]):
        np.linalg.cholesky(covs[k])


def test_degenerate_cluster_uses_identity_fallback_and_logs_warning(caplog):
    """A cluster with 1 point must use identity fallback and emit a WARNING."""
    # Construct data where one cluster will have a single point by using
    # extreme separation so KMeans assigns one centroid to the outlier.
    rng = np.random.default_rng(5)
    # One extreme outlier far from the main cluster.
    outlier = rng.standard_normal((1, 3)) + 1000.0
    main = rng.standard_normal((200, 3))
    X = np.vstack([outlier, main])
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    with caplog.at_level(logging.WARNING, logger="regime_ml"):
        _, covs, _ = initialise_emissions(df, n_clusters=2, random_state=0)
    # Must still be PD regardless of which cluster is degenerate.
    for k in range(covs.shape[0]):
        np.linalg.cholesky(covs[k])
    assert any("identity fallback" in r.message for r in caplog.records)


def test_diag_covariance_type_returns_correct_shape():
    """covariance_type='diag' must return shape (n_clusters, n_features)."""
    df = _make_df(100, d=5)
    _, covs, _ = initialise_emissions(
        df, n_clusters=2, random_state=42, covariance_type="diag"
    )
    assert covs.shape == (2, 5)


def test_diag_covariance_values_are_positive():
    """Diagonal covariance entries must all be positive."""
    df = _make_df(100, d=4)
    _, covs, _ = initialise_emissions(
        df, n_clusters=2, random_state=42, covariance_type="diag"
    )
    assert (covs > 0).all()
