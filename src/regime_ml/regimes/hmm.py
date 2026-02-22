import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import BaseRegimeDetector

logger = logging.getLogger(__name__)

def initialise_emissions(
    df_train: pd.DataFrame,
    n_clusters: int,
    random_state: Optional[int] = None,
    covariance_type: str = 'full',
    scale_features: bool = True,
    train_end_date: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Initialize HMM emission parameters using KMeans clustering.

    Computes cluster means and covariances from k-means clustering on training data.
    These can be used to initialize HMM emission distributions.

    Args:
        df_train: Training features DataFrame (n_samples, n_features).
                  Must contain only in-sample data. If df_train has a DatetimeIndex
                  and train_end_date is provided, this is enforced with a ValueError.
        n_clusters: Number of clusters (should match n_regimes)
        random_state: Random seed for k-means initialization
        covariance_type: 'full' or 'diag' - format of returned covariance matrices
        scale_features: Whether to standardize features before clustering (recommended)
        train_end_date: Optional upper bound for the training period. If df_train has a
                        DatetimeIndex and any date exceeds this value, a ValueError is
                        raised to prevent OOS data from leaking into the scaler fit.

    Returns:
        Tuple of (means, covariances, scaler):
        - means: Cluster means array (n_clusters, n_features)
        - covariances: Covariance matrices array, shape depends on covariance_type
        - scaler: Fitted StandardScaler for transforming new data.
                  The scaler carries a ``_regime_ml_train_date_range`` attribute
                  with (min_date, max_date) of the training set for audit purposes.
    """
    # Enforce IS-only scaler fitting
    _train_date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    if isinstance(df_train.index, pd.DatetimeIndex):
        train_min = df_train.index.min()
        train_max = df_train.index.max()
        _train_date_range = (train_min, train_max)
        logger.info(
            "initialise_emissions: %d rows, %s to %s",
            len(df_train), train_min.date(), train_max.date()
        )
        if train_end_date is not None and train_max > train_end_date:
            raise ValueError(
                "df_train contains out-of-sample dates: max date in df_train is %s "
                "but train_end_date is %s. Pass only in-sample data to "
                "initialise_emissions()." % (train_max.date(), train_end_date.date())
            )
        if train_end_date is not None:
            logger.info(
                "initialise_emissions: IS boundary respected (max=%s <= train_end_date=%s)",
                train_max.date(), train_end_date.date()
            )
    else:
        logger.info(
            "initialise_emissions: df_train has no DatetimeIndex; "
            "IS boundary cannot be validated — pass train_end_date with a DatetimeIndex to enforce."
        )

    # Convert to numpy array
    X_train = df_train.values
    n_features = X_train.shape[1]

    # Scale features (important for k-means distance-based clustering)
    scaler = StandardScaler()
    if scale_features:
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = X_train.copy()
        # Still fit scaler for consistency (identity transform)
        scaler.fit(X_train)

    # Attach training date range to scaler for audit/inspection
    scaler._regime_ml_train_date_range = _train_date_range  # type: ignore[attr-defined]
    
    # Fit k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_train_scaled)
    
    # Get cluster means (from k-means centers)
    means = kmeans.cluster_centers_  # Shape: (n_clusters, n_features)
    
    # Compute cluster covariances
    covariances = []
    
    for cluster_id in range(n_clusters):
        # Get points assigned to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_points = X_train_scaled[cluster_mask]
        
        if len(cluster_points) < 2:
            # Not enough points for covariance - use identity matrix
            cov = np.eye(n_features) * 1e-6
        else:
            # Compute sample covariance matrix (unbiased, divide by n-1)
            cov = np.cov(cluster_points.T, ddof=1)
            
            # Ensure positive semi-definite (numerical stability)
            cov = (cov + cov.T) / 2  # Ensure symmetry
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)  # Ensure positive eigenvalues
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Cholesky jitter: guarantee numerical positive-definiteness before
            # hmmlearn's internal Cholesky decomposition during EM.
            cov += 1e-6 * np.eye(n_features)
        
        # Handle NaN/Inf values
        cov = np.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
        covariances.append(cov)
    
    # Format covariances based on requested type
    if covariance_type == 'full':
        # Stack into (n_clusters, n_features, n_features) array
        covs_array = np.stack(covariances, axis=0)
        return means, covs_array, scaler
    elif covariance_type == 'diag':
        # Extract diagonal elements: (n_clusters, n_features)
        covs_array = np.stack([np.diagonal(cov) for cov in covariances], axis=0)
        return means, covs_array, scaler
    else:
        raise ValueError(f"Unsupported covariance type: {covariance_type}. Use 'full' or 'diag'.")

def initialise_transitions(n_regimes: int, p_stay: float) -> np.ndarray:
    """
    Initialize HMM transition matrix with uniform off-diagonal probabilities.
    
    Creates a transition matrix where diagonal elements (staying in same regime)
    have probability p_stay, and off-diagonal elements (switching regimes) are
    uniformly distributed with probability (1 - p_stay) / (n_regimes - 1).
    
    Args:
        n_regimes: Number of regimes/states
        p_stay: Probability of staying in the same regime (diagonal elements)
    
    Returns:
        Transition matrix (n_regimes, n_regimes) with rows summing to 1
    """
    if not 0 <= p_stay <= 1:
        raise ValueError(f"p_stay must be between 0 and 1, got {p_stay}")
    
    transmat = (
        np.eye(n_regimes) * p_stay + 
        (np.ones((n_regimes, n_regimes)) - np.eye(n_regimes)) * (1 - p_stay) / (n_regimes - 1)
    )
    return transmat
    
def initialise_probabilities(n_regimes: int) -> np.ndarray:
    """
    Initialize uniform initial state probabilities.
    
    All regimes have equal probability of being the initial state.
    
    Args:
        n_regimes: Number of regimes/states
    
    Returns:
        Initial state probability vector (n_regimes,) summing to 1
    """
    return np.ones(n_regimes) / n_regimes

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
        n_iter: int = 500,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        startprob: Optional[np.ndarray] = None,
        transmat: Optional[np.ndarray] = None,
        means: Optional[np.ndarray] = None,
        covars: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        scaler: Optional[StandardScaler] = None,
        init_params: str = '',
        **kwargs
    ):
        """
        Initialize HMM regime detector.
        
        Args:
            n_regimes: Number of hidden states (regimes)
            covariance_type: 'full', 'diag', 'spherical', 'tied'
            n_iter: Maximum EM iterations
            tol: Convergence tolerance
            random_state: Random seed. None uses an unpredictable seed — useful for
                detecting label-switching instability across runs. Set a fixed integer
                for reproducible research.
            startprob: Optional initial state probabilities (n_regimes,)
            transmat: Optional transition matrix (n_regimes, n_regimes)
            means: Optional emission means (n_regimes, n_features)
            covars: Optional emission covariances (format depends on covariance_type)
            init_params: Parameters to initialize (empty string means use provided values)
            **kwargs: Additional arguments passed to GaussianHMM
        """
        super().__init__(n_regimes=n_regimes)
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        self.startprob = startprob
        self.transmat = transmat
        self.means = means
        self.covars = covars
        
        self.feature_names = feature_names
        self.state_labels: Dict[int, str] = {}
    
        self.scaler = scaler
        
        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            init_params=init_params,
            **kwargs
        )

    def fit(self, X: np.ndarray, **kwargs) -> 'HMMRegimeDetector':
        """
        Fit HMM to features.
        
        Args:
            X: Features array (T, n_features)
            **kwargs: Additional arguments passed to GaussianHMM.fit()
            
        Returns:
            self (fitted model)
        """
        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (T, n_features), got shape {X.shape}")
        
        # Set initialization parameters if provided
        if self.startprob is not None:
            self.model.startprob_ = self.startprob
        if self.transmat is not None:
            self.model.transmat_ = self.transmat
        if self.means is not None:
            self.model.means_ = self.means
        if self.covars is not None:
            self.model.covars_ = self.covars

        # Fit HMM
        self.model.fit(X, **kwargs)
        monitor = self.model.monitor_
        # hmmlearn 0.3.x sets converged=True when iter==n_iter (hit limit) OR when
        # LL delta < tol. We only want to warn when the LL delta criterion was NOT met,
        # i.e. the model stopped because it ran out of iterations, not because it converged.
        ll_converged = (
            len(monitor.history) >= 2
            and monitor.history[-1] - monitor.history[-2] < monitor.tol
        )
        if not ll_converged:
            delta = (
                monitor.history[-1] - monitor.history[-2]
                if len(monitor.history) >= 2
                else float("nan")
            )
            logger.warning(
                "HMM did not converge after %d iterations "
                "(final LL delta: %.6f, tol=%.6f). "
                "Consider increasing n_iter or relaxing tol in regime_config.yaml.",
                self.n_iter, delta, self.tol,
            )
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime labels using Viterbi algorithm.
        
        Args:
            X: Features array (T, n_features)
            
        Returns:
            Regime labels array (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (T, n_features), got shape {X.shape}")
        
        return self.model.predict(X)
    
    def smooth_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Smoothed state posteriors, P(z_t | x_1:T). NOT Causal, as uses past and future data in forward-backward algorithm.
        
        Args:
            X: Features array (T, n_features)
            
        Returns:
            Regime probabilities array (T, n_regimes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (T, n_features), got shape {X.shape}")
        # hmm returns posterior probabilities from forward-backward algorithm, so NOT causal.
        return self.model.predict_proba(X)

    def filter_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Filtered state probabilities P(z_t | x_1:t). Causal, as uses only past data in forward recursion.
        
        Args:
            X: Features array (T, n_features)
            
        Returns:
            Regime probabilities array (T, n_regimes)
        """
        def _log_sum_exp(a: np.ndarray, axis=None) -> np.ndarray:
            """Computes the sum of the exponents of each element in the array a"""
            a_max = np.max(a, axis=axis, keepdims=True)
            out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
            return np.squeeze(out, axis=axis)
        
        def _log_gaussian_pdf_full(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
            """
            Computes the log emission likelihood log p(x(t | z_t = k))

            X: (T, d), means (n_regimes, d), covars (n_regimes, d, d)
            returns loglik (T, n_regimes) with log N(X_t | mean_k, covar_k)
            """
            T, d = X.shape
            n_regimes = means.shape[0]
            loglik = np.empty((T, n_regimes), dtype=float)
            const = -0.5 * d * np.log(2 * np.pi)

            for k in range(n_regimes):
                C = covars[k]
                # add a small jitter to ensure positive semi-definitivity
                C = C + 1e-8 * np.eye(d)
                # Cholesky decomposition of the covariance matrix C
                L = np.linalg.cholesky(C)
                Xm = X - means[k]
                y = np.linalg.solve(L, Xm.T).T # (T, d)
                # Magnitude squared of the vector y
                quad = np.sum(y*y, axis=1)
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                loglik[:, k] = const - 0.5 * (quad + logdet)
                
            return loglik

        def _log_gaussian_pdf_diag(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
            """
            X: (T,d), means: (K,d), covars: (K,d) variances
            returns (T,K)
            """
            T, d = X.shape
            n_regimes = means.shape[0]
            loglik = np.empty((T, n_regimes), dtype=float)
            const = -0.5 * d * np.log(2 * np.pi)
            for k in range(n_regimes):
                var = covars[k]
                var = np.maximum(var, 1e-8)
                Xm = X - means[k]
                quad = np.sum((Xm * Xm) / var, axis=1)
                logdet = np.sum(np.log(var))
                loglik[:, k] = const - 0.5 * (logdet + quad)
            return loglik

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Pull HMM parameters from fitted model
        A = np.asarray(self.model.transmat_, dtype=float)   # (K,K) - transition matrix
        pi = np.asarray(self.model.startprob_, dtype=float) # (K,)  - initial state probabilities
        means = np.asarray(self.model.means_, dtype=float)  # (K,d) - emission means

        n_regimes, d = means.shape
        assert A.shape == (n_regimes, n_regimes)
        assert pi.shape == (n_regimes,)
        assert np.allclose(pi.sum(), 1.0, atol=1e-6)
        assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)

        # Pull emission covariances from fitted model (differerent functions for full and diagonal covariance matrices)
        if self.model.covariance_type == 'full':
            covars = np.asarray(self.model.covars_, dtype=float)    # (K,d,d)
            # emission log-likelihoods for full covariance matrices
            logB = _log_gaussian_pdf_full(X, means, covars)         # (T,K)
        elif self.model.covariance_type == 'diag':
            covars = np.asarray(self.model.covars_, dtype=float)
            # check what dimension hmmlearn returns for covariance matrices
            if covars.ndim == 3:
                covars = np.diagonal(covars, axis1=1, axis2=2)      # (K,d)
            #emission log-likelihoods for diagonal covariance matrices
            logB = _log_gaussian_pdf_diag(X, means, covars)         # (T,K)
        else:
            raise NotImplementedError(f"Unsupported covariance type: {self.model.covariance_type}. Use 'full' or 'diag'.")

        if not np.isfinite(logB).all():
            raise ValueError("Non-finite emission log-likelihoods (logB). Check covars / scaling.")

        T, n_regimes = logB.shape

        # convert transitions and initial state probabilities to log space
        logA = np.log(np.maximum(A, 1e-300))
        logpi = np.log(np.maximum(pi, 1e-300))
        
        # forward recursion in log space
        log_alpha = np.zeros((T, n_regimes))

        # t=0, bayes rule + normalisation
        log_alpha[0] = logpi + logB[0]
        log_alpha[0] -= _log_sum_exp(log_alpha[0])

        # t=1, ..., T-1
        for t in range(1, T):
            # log_pred[k] = log sum_i exp(log_alpha[t-1, i] + logA[i, k])
            log_pred = _log_sum_exp(log_alpha[t-1][:, None] + logA, axis=0) # (K,)

            log_alpha[t] = logB[t] + log_pred
            log_alpha[t] -= _log_sum_exp(log_alpha[t])

        return np.exp(log_alpha)

    def forecast_n_steps(self,
        proba: np.ndarray,
        n: int
    ) -> np.ndarray:
        """
        n-step ahead regime forecast, gives a forecast for the next n regimes for every time t

        alpha_t: (n_regimes), filtered regime probabilities at time t
        A: (n_regimes, n_regimes), transition matrix
        n: forecast horizon
        
        returns: (n_regimes, n_steps)
        """
        A = np.asarray(self.model.transmat_, dtype=float)
        A_n = np.linalg.matrix_power(A, n)
        return proba @ A_n

    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of data under the fitted model.
        
        Args:
            X: Features array (T, n_features)
            
        Returns:
            Log-likelihood score
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (n_samples, n_features), got shape {X.shape}")
        return self.model.score(X)
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition probability matrix.
        
        Returns:
            Transition matrix (n_regimes, n_regimes) where entry (i,j) is
            probability of transitioning from regime i to regime j
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.transmat_
    
    def get_regime_means(self) -> np.ndarray:
        """
        Get mean feature values for each regime.
        
        Returns:
            Means array (n_regimes, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.means_
    
    def get_regime_covariances(self) -> np.ndarray:
        """
        Get covariance matrices for each regime.
        
        Returns:
            Covariance array, shape depends on covariance_type:
            - 'full': (n_regimes, n_features, n_features)
            - 'diag': (n_regimes, n_features)
            - 'spherical': (n_regimes,)
            - 'tied': (n_features, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.covars_ # type: ignore
    
    def save(self, path: Path) -> None:
        """
        Save fitted model to pickle file.
        
        Args:
            path: File path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> 'HMMRegimeDetector':
        """
        Load fitted model from pickle file.
        
        Args:
            path: File path to load model from
            
        Returns:
            Loaded HMMRegimeDetector instance
        """
        with open(path, 'rb') as f:
            return pickle.load(f)