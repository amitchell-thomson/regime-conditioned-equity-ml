"""Tests for HMMRegimeDetector.save() / load() — JSON/numpy parameter format (Phase 7.2)."""

import json

import numpy as np
import pandas as pd
import pytest

from regime_ml.regimes.hmm import (
    HMMRegimeDetector,
    initialise_emissions,
    initialise_probabilities,
    initialise_transitions,
)


def _make_df(n: int = 300, d: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-01", periods=n)
    return pd.DataFrame(
        rng.standard_normal((n, d)), index=idx, columns=[f"f{i}" for i in range(d)]
    )


def _fitted_detector(n: int = 300, d: int = 3, K: int = 3, cov: str = "diag"):
    df = _make_df(n=n, d=d)
    means, covs, scaler = initialise_emissions(
        df, n_clusters=K, random_state=0, covariance_type=cov
    )
    det = HMMRegimeDetector(
        n_regimes=K,
        covariance_type=cov,
        random_state=0,
        means=means,
        covars=covs,
        transmat=initialise_transitions(K, p_stay=0.9),
        startprob=initialise_probabilities(K),
    )
    X = scaler.transform(df.values)
    det.fit(X)
    return det, X, scaler


class TestHMMSaveLoad:
    def test_save_creates_directory(self, tmp_path):
        """save() creates the target directory."""
        det, X, _ = _fitted_detector()
        save_dir = tmp_path / "model"
        det.save(save_dir)
        assert save_dir.is_dir()

    def test_save_creates_expected_files(self, tmp_path):
        """save() writes metadata.json and all required .npy files."""
        det, X, _ = _fitted_detector()
        save_dir = tmp_path / "model"
        det.save(save_dir)

        expected = {
            "metadata.json",
            "transmat.npy",
            "startprob.npy",
            "means.npy",
            "covars.npy",
        }
        actual = {f.name for f in save_dir.iterdir()}
        assert expected.issubset(actual), f"Missing files: {expected - actual}"

    def test_metadata_json_contains_expected_keys(self, tmp_path):
        """metadata.json must contain n_regimes, covariance_type, n_iter, tol."""
        det, X, _ = _fitted_detector(K=3)
        save_dir = tmp_path / "model"
        det.save(save_dir)

        with open(save_dir / "metadata.json") as fh:
            meta = json.load(fh)

        assert meta["n_regimes"] == 3
        assert meta["covariance_type"] == "diag"
        assert "n_iter" in meta
        assert "tol" in meta

    def test_load_roundtrip_predict_identical(self, tmp_path):
        """load() produces a detector whose predict() output matches the original."""
        det, X, _ = _fitted_detector(K=3)
        labels_before = det.predict(X)

        save_dir = tmp_path / "model"
        det.save(save_dir)
        loaded = HMMRegimeDetector.load(save_dir)

        labels_after = loaded.predict(X)
        np.testing.assert_array_equal(labels_before, labels_after)

    def test_load_roundtrip_filter_proba_close(self, tmp_path):
        """filter_proba() output from a loaded model matches original within tolerance."""
        det, X, _ = _fitted_detector(K=3)
        proba_before = det.filter_proba(X)

        save_dir = tmp_path / "model"
        det.save(save_dir)
        loaded = HMMRegimeDetector.load(save_dir)

        proba_after = loaded.filter_proba(X)
        np.testing.assert_allclose(proba_before, proba_after, rtol=1e-5, atol=1e-8)

    def test_load_raises_on_missing_directory(self, tmp_path):
        """load() raises FileNotFoundError for a nonexistent path."""
        with pytest.raises(FileNotFoundError):
            HMMRegimeDetector.load(tmp_path / "nonexistent")

    def test_save_raises_if_not_fitted(self, tmp_path):
        """save() raises ValueError if the model has not been fitted."""
        det = HMMRegimeDetector(n_regimes=3)
        with pytest.raises(ValueError, match="unfitted"):
            det.save(tmp_path / "model")

    def test_loaded_model_is_fitted(self, tmp_path):
        """Loaded detector has is_fitted=True."""
        det, X, _ = _fitted_detector()
        save_dir = tmp_path / "model"
        det.save(save_dir)
        loaded = HMMRegimeDetector.load(save_dir)
        assert loaded.is_fitted is True

    def test_no_pickle_import_in_hmm_source(self):
        """Confirm 'import pickle' has been removed from hmm.py."""
        from pathlib import Path

        source = (
            Path(__file__).parent.parent / "src" / "regime_ml" / "regimes" / "hmm.py"
        ).read_text()
        assert "import pickle" not in source, (
            "hmm.py still contains 'import pickle' — pickle must be replaced with JSON/numpy"
        )
