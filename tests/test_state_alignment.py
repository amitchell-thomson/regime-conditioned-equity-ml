"""Tests for align_states() and permute_detector() (item 3.4)."""

import numpy as np
import pandas as pd
import pytest

from regime_ml.regimes.hmm import (
    HMMRegimeDetector,
    align_states,
    initialise_emissions,
    initialise_probabilities,
    initialise_transitions,
    permute_detector,
)


def _fit_detector(
    seed: int = 0,
    n_regimes: int = 3,
    n_samples: int = 300,
    n_features: int = 3,
) -> HMMRegimeDetector:
    """Fit a minimal HMMRegimeDetector on synthetic Gaussian data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    means, covs, scaler = initialise_emissions(
        df, n_clusters=n_regimes, random_state=seed
    )
    det = HMMRegimeDetector(
        n_regimes=n_regimes,
        random_state=seed,
        means=means,
        covars=covs,
        transmat=initialise_transitions(n_regimes, p_stay=0.9),
        startprob=initialise_probabilities(n_regimes),
    )
    det.fit(scaler.transform(X))
    return det


# ---------------------------------------------------------------------------
# align_states() tests
# ---------------------------------------------------------------------------


def test_align_states_returns_valid_permutation():
    """align_states must return a valid permutation array of length K."""
    ref = _fit_detector(seed=0)
    cand = _fit_detector(seed=1)
    perm = align_states(ref, cand)
    K = ref.n_regimes
    assert perm.shape == (K,)
    assert set(perm.tolist()) == set(range(K))


def test_align_states_identity_on_same_model():
    """Aligning a model to itself should return the identity permutation."""
    ref = _fit_detector(seed=42)
    perm = align_states(ref, ref)
    np.testing.assert_array_equal(perm, np.arange(ref.n_regimes))


def test_align_states_unfitted_reference_raises():
    """align_states must raise ValueError if the reference detector is unfitted."""
    fitted = _fit_detector(seed=0)
    unfitted = HMMRegimeDetector(n_regimes=3)
    with pytest.raises(ValueError, match="fitted"):
        align_states(unfitted, fitted)


def test_align_states_unfitted_candidate_raises():
    """align_states must raise ValueError if the candidate detector is unfitted."""
    fitted = _fit_detector(seed=0)
    unfitted = HMMRegimeDetector(n_regimes=3)
    with pytest.raises(ValueError, match="fitted"):
        align_states(fitted, unfitted)


def test_align_states_n_regimes_mismatch_raises():
    """align_states must raise ValueError when n_regimes differ."""
    ref = _fit_detector(seed=0, n_regimes=2)
    cand = _fit_detector(seed=1, n_regimes=3)
    with pytest.raises(ValueError, match="mismatch"):
        align_states(ref, cand)


def test_align_states_non_full_covariance_raises():
    """align_states must raise NotImplementedError for non-full covariance types."""
    # Build two detectors with covariance_type='diag'
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    means, covs, scaler = initialise_emissions(
        df, n_clusters=2, random_state=0, covariance_type="diag"
    )
    det = HMMRegimeDetector(
        n_regimes=2,
        covariance_type="diag",
        means=means,
        covars=covs,
        transmat=initialise_transitions(2, p_stay=0.9),
        startprob=initialise_probabilities(2),
    )
    det.fit(scaler.transform(X))
    with pytest.raises(NotImplementedError):
        align_states(det, det)


# ---------------------------------------------------------------------------
# permute_detector() tests
# ---------------------------------------------------------------------------


def test_permute_detector_startprob_sums_to_one():
    """Permuted startprob must still sum to 1."""
    ref = _fit_detector(seed=0)
    cand = _fit_detector(seed=1)
    perm = align_states(ref, cand)
    permuted = permute_detector(cand, perm)
    np.testing.assert_allclose(permuted.model.startprob_.sum(), 1.0, atol=1e-9)


def test_permute_detector_transmat_rows_sum_to_one():
    """Each row of the permuted transition matrix must sum to 1."""
    ref = _fit_detector(seed=0)
    cand = _fit_detector(seed=1)
    perm = align_states(ref, cand)
    permuted = permute_detector(cand, perm)
    np.testing.assert_allclose(
        permuted.model.transmat_.sum(axis=1),
        np.ones(ref.n_regimes),
        atol=1e-9,
    )


def test_permute_detector_means_reordered_correctly():
    """permute_detector must apply the same permutation to means."""
    ref = _fit_detector(seed=0)
    cand = _fit_detector(seed=1)
    perm = align_states(ref, cand)
    permuted = permute_detector(cand, perm)
    for new_k in range(ref.n_regimes):
        old_k = int(np.where(perm == new_k)[0][0])
        np.testing.assert_allclose(
            permuted.model.means_[new_k],
            cand.model.means_[old_k],
            atol=1e-10,
        )


def test_permute_detector_does_not_mutate_input():
    """permute_detector must not modify the original detector's means."""
    ref = _fit_detector(seed=0)
    cand = _fit_detector(seed=1)
    original_means = cand.model.means_.copy()
    perm = align_states(ref, cand)
    _ = permute_detector(cand, perm)
    np.testing.assert_array_equal(cand.model.means_, original_means)


def test_permute_detector_is_fitted():
    """Returned detector must have is_fitted == True."""
    ref = _fit_detector(seed=0)
    cand = _fit_detector(seed=1)
    perm = align_states(ref, cand)
    permuted = permute_detector(cand, perm)
    assert permuted.is_fitted


def test_permute_detector_invalid_perm_raises():
    """permute_detector must raise ValueError for arrays with duplicate indices."""
    det = _fit_detector(seed=0, n_regimes=3)
    with pytest.raises(ValueError, match="permutation"):
        permute_detector(det, np.array([0, 0, 2]))


def test_permute_detector_wrong_length_raises():
    """permute_detector must raise ValueError if perm length != n_regimes."""
    det = _fit_detector(seed=0, n_regimes=3)
    with pytest.raises(ValueError, match="permutation"):
        permute_detector(det, np.array([0, 1]))
