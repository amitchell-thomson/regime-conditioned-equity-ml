import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from hmmlearn import hmm
from typing import Optional

from .base import BaseRegimeDetector

class HMMRegimeDetector(BaseRegimeDetector):
    """
    Hidden Markov Model for regime detection.
    
    Uses Gaussian emissions with full covariance matrices.
    Suitable for 3-8 features, 2-5 regimes.
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = 'full',
        n_iter: int = 1000,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize HMM regime detector.
        
        Args:
            n_regimes: Number of hidden states (regimes)
            covariance_type: 'full', 'diag', 'spherical', 'tied'
            n_iter: Maximum EM iterations
            random_state: Random seed for reproducibility
        """
        super().__init__(n_regimes=n_regimes)
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            **kwargs
        )
        
    def fit(self, X: pd.DataFrame, **kwargs) -> 'HMMRegimeDetector':
        """
        Fit HMM to features.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            self (fitted)
        """
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Fit HMM
        X_array = X.values
        self.model.fit(X_array, **kwargs)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime labels using Viterbi algorithm."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_array = X[self.feature_names].values
        return self.model.predict(X_array)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get regime probabilities using forward algorithm."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_array = X[self.feature_names].values
        return self.model.predict_proba(X_array)
    
    def score(self, X: pd.DataFrame) -> float:
        """Compute log-likelihood of data."""
        X_array = X[self.feature_names].values
        return self.model.score(X_array)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get regime transition probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.transmat_
    
    def get_regime_means(self) -> np.ndarray:
        """Get mean feature values for each regime."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.means_
    
    def save(self, path: Path) -> None:
        """Save model to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> 'HMMRegimeDetector':
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)