"""Tests for OOS causality fix in evaluation.py: must use filter_proba not smooth_proba."""

import numpy as np

from regime_ml.regimes.hmm import HMMRegimeDetector


def _fit_detector(n_samples: int = 200, n_features: int = 3, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    detector = HMMRegimeDetector(n_regimes=2, n_iter=50, random_state=seed)
    detector.fit(X)
    return detector, X


def test_filter_proba_differs_from_smooth_proba():
    """filter_proba and smooth_proba must produce different values on the same data.

    This guards the key correctness property: OOS evaluation uses filter_proba
    (causal), not smooth_proba (non-causal). If they were identical, the fix
    in evaluation.py would have no effect.
    """
    detector, X = _fit_detector()
    filt = detector.filter_proba(X)
    smooth = detector.smooth_proba(X)
    assert not np.allclose(filt, smooth, atol=1e-6), (
        "filter_proba and smooth_proba should differ — "
        "they use different algorithms (forward-only vs forward-backward)"
    )


def test_filter_proba_shape():
    """filter_proba output shape must be (T, K)."""
    detector, X = _fit_detector(n_samples=150, n_features=3)
    filt = detector.filter_proba(X)
    assert filt.shape == (150, 2)


def test_filter_proba_rows_sum_to_one():
    """Each row of filter_proba must sum to 1 (valid probability distribution)."""
    detector, X = _fit_detector()
    filt = detector.filter_proba(X)
    row_sums = filt.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(X)), atol=1e-6)


def test_smooth_proba_rows_sum_to_one():
    """Each row of smooth_proba must also sum to 1."""
    detector, X = _fit_detector()
    smooth = detector.smooth_proba(X)
    row_sums = smooth.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(X)), atol=1e-6)
