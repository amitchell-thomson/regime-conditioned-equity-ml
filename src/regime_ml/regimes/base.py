from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

class BaseRegimeDetector(ABC):
    """
    Abstract base class for all regime detection models.
    
    All regime detectors must implement:
    - fit(): Learn regimes from data
    - predict(): Assign regimes to new data
    - predict_proba(): Get regime probabilities
    - save/load: Persist trained models
    """
    
    def __init__(self, n_regimes: int = 4, **kwargs):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
            **kwargs: Model-specific parameters
        """
        self.n_regimes = n_regimes
        self.is_fitted = False
        self.feature_names = None
        self.params = kwargs
        
    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> 'BaseRegimeDetector':
        """
        Fit regime model to data.
        
        Args:
            X: Features array (n_samples, n_features)
            
        Returns:
            self (fitted model)
        """
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for data.
        
        Args:
            X: Features array (n_samples, n_features)
            
        Returns:
            Regime labels (n_samples,)
        """
        ...
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.
        
        Args:
            X: Features array (n_samples, n_features)
            
        Returns:
            Regime probabilities (n_samples, n_regimes)
        """
        ...
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method: fit and predict in one call."""
        self.fit(X)
        return self.predict(X)
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save fitted model to disk."""
        ...
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> 'BaseRegimeDetector':
        """Load fitted model from disk."""
        ...
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_regimes': self.n_regimes,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            **self.params
        }
    
    def __repr__(self) -> str:
        params_str = ', '.join([f'{k}={v}' for k, v in self.get_params().items()])
        return f"{self.__class__.__name__}({params_str})"