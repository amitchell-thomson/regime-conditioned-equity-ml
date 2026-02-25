import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
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
            # Degenerate cluster: not enough points for shrinkage estimation.
            cov = np.eye(n_features) * 1e-6
            logger.warning(
                "initialise_emissions: cluster %d has %d point(s); using identity fallback.",
                cluster_id, len(cluster_points),
            )
        else:
            # Ledoit-Wolf shrinkage: guarantees PSD for any n_samples >= 2,
            # including the rank-deficient case (n_points < n_features).
            lw = LedoitWolf()
            lw.fit(cluster_points)
            cov = lw.covariance_

        # Symmetry guard against float drift (LW should already be symmetric).
        cov = (cov + cov.T) / 2
        # Cholesky jitter: guarantee strict PD before hmmlearn's internal decomp.
        cov += 1e-6 * np.eye(n_features)
        # Sanitise any residual non-finite values from extreme cluster configs.
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


def _kl_gaussian(
    mu_p: np.ndarray,
    cov_p: np.ndarray,
    mu_q: np.ndarray,
    cov_q: np.ndarray,
    jitter: float = 1e-8,
) -> float:
    """
    KL divergence KL(p || q) for multivariate Gaussians.

    KL(p||q) = 0.5 * [log|Σ_q| - log|Σ_p| + tr(Σ_q^{-1} Σ_p)
                       + (μ_p - μ_q)^T Σ_q^{-1} (μ_p - μ_q) - d]

    Private helper for align_states(). Returns np.inf on numerical failure.
    """
    d = mu_p.shape[0]
    Cq = cov_q + jitter * np.eye(d)
    Cp = cov_p + jitter * np.eye(d)
    try:
        Cq_inv = np.linalg.inv(Cq)
    except np.linalg.LinAlgError:
        return np.inf
    sign_q, logdet_q = np.linalg.slogdet(Cq)
    sign_p, logdet_p = np.linalg.slogdet(Cp)
    if sign_q <= 0 or sign_p <= 0:
        return np.inf
    diff = mu_p - mu_q
    kl = 0.5 * (
        logdet_q - logdet_p
        + np.trace(Cq_inv @ Cp)
        + float(diff @ Cq_inv @ diff)
        - d
    )
    return float(max(kl, 0.0))


def align_states(
    reference_detector: "HMMRegimeDetector",
    candidate_detector: "HMMRegimeDetector",
) -> np.ndarray:
    """
    Compute the optimal permutation aligning candidate states to reference states.

    Builds a (K, K) cost matrix where cost[i, j] = KL(candidate_i || reference_j)
    and solves the linear assignment problem to find the minimum-cost bijection.

    Args:
        reference_detector: Fitted HMMRegimeDetector defining the target labelling.
        candidate_detector: Fitted HMMRegimeDetector to be realigned.

    Returns:
        perm (np.ndarray, shape (K,)): perm[candidate_state] = reference_state.
        Pass to permute_detector() to produce a relabelled candidate.

    Raises:
        ValueError: If either detector is unfitted or n_regimes differ.
        NotImplementedError: If covariance_type != 'full' (only full supported).
    """
    if not reference_detector.is_fitted or not candidate_detector.is_fitted:
        raise ValueError(
            "Both detectors must be fitted before calling align_states()."
        )
    if reference_detector.n_regimes != candidate_detector.n_regimes:
        raise ValueError(
            f"n_regimes mismatch: reference has {reference_detector.n_regimes}, "
            f"candidate has {candidate_detector.n_regimes}."
        )
    if (reference_detector.covariance_type != "full"
            or candidate_detector.covariance_type != "full"):
        raise NotImplementedError(
            "align_states() only supports covariance_type='full'. "
            f"Got reference={reference_detector.covariance_type!r}, "
            f"candidate={candidate_detector.covariance_type!r}."
        )

    K = reference_detector.n_regimes
    mu_ref   = reference_detector.model.means_    # (K, d)
    cov_ref  = reference_detector.model.covars_   # (K, d, d)
    mu_cand  = candidate_detector.model.means_    # (K, d)
    cov_cand = candidate_detector.model.covars_   # (K, d, d)

    cost = np.array([
        [_kl_gaussian(mu_cand[i], cov_cand[i], mu_ref[j], cov_ref[j]) for j in range(K)]
        for i in range(K)
    ])
    _, col_ind = linear_sum_assignment(cost)
    return col_ind  # perm[candidate_state] = reference_state


def permute_detector(
    detector: "HMMRegimeDetector",
    perm: np.ndarray,
) -> "HMMRegimeDetector":
    """
    Return a new HMMRegimeDetector with state parameters reordered by perm.

    All arrays are deep-copied; the input detector is not mutated.

    Args:
        detector: A fitted HMMRegimeDetector.
        perm:     1-D int array of length K; perm[old_state] = new_state.
                  Typically the output of align_states().

    Returns:
        New fitted HMMRegimeDetector with reordered means, covars, transmat,
        and startprob.

    Raises:
        ValueError: If perm is not a valid permutation of {0, ..., K-1}.
    """
    import copy

    K = detector.n_regimes
    if perm.shape != (K,) or set(perm.tolist()) != set(range(K)):
        raise ValueError(
            f"perm must be a valid permutation of {{0,...,{K - 1}}}, got {perm}."
        )

    new_det = copy.deepcopy(detector)
    new_det.model.means_    = detector.model.means_[perm]
    new_det.model.covars_   = detector.model.covars_[perm]
    A = detector.model.transmat_
    new_det.model.transmat_ = A[np.ix_(perm, perm)]
    new_det.model.startprob_ = detector.model.startprob_[perm]
    new_det.is_fitted = True
    return new_det


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
            Log-likelihood score (per sample)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (n_samples, n_features), got shape {X.shape}")
        return self.model.score(X)

    def _n_params(self) -> int:
        """Number of free parameters in the fitted HMM.

        Counts:
          - Initial state probs: K-1 (sums to 1)
          - Transition matrix: K*(K-1) (each row sums to 1)
          - Emission means: K*D
          - Emission covariances: depends on covariance_type
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        K = self.n_regimes
        D = int(self.model.means_.shape[1])
        n_base = (K - 1) + K * (K - 1) + K * D
        cov_type = self.covariance_type
        if cov_type == "diag":
            n_cov = K * D
        elif cov_type == "full":
            n_cov = K * D * (D + 1) // 2
        elif cov_type == "spherical":
            n_cov = K
        elif cov_type == "tied":
            n_cov = D * (D + 1) // 2
        else:
            n_cov = K * D  # fallback for unknown types
        return n_base + n_cov

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion.

        BIC = -2 * log_likelihood + n_params * log(n_samples).
        Lower BIC indicates a better model, penalising unnecessary complexity.

        Args:
            X: Features array (T, n_features) — must be the same scaled data
               used for fitting.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        n = len(X)
        log_lik = self.score(X) * n  # score() returns per-sample log-likelihood
        return float(-2.0 * log_lik + self._n_params() * np.log(n))

    def aic(self, X: np.ndarray) -> float:
        """Akaike Information Criterion.

        AIC = -2 * log_likelihood + 2 * n_params.
        Lower AIC indicates a better model.

        Args:
            X: Features array (T, n_features) — must be the same scaled data
               used for fitting.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        n = len(X)
        log_lik = self.score(X) * n
        return float(-2.0 * log_lik + 2.0 * self._n_params())
    
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


def fit_best_of_n_seeds(
    df_train: pd.DataFrame,
    n_regimes: int,
    n_init: int,
    train_end_date: Optional[pd.Timestamp] = None,
    covariance_type: str = "full",
    n_iter: int = 500,
    tol: float = 1e-4,
    p_stay: float = 0.9,
    min_regime_share: float = 0.03,
    feature_names: Optional[list[str]] = None,
) -> Tuple[HMMRegimeDetector, StandardScaler]:
    """
    Fit HMM across n_init seeds and return the best-performing model.

    Seeds are derived as range(n_init) for full reproducibility — all
    parameters must come from config, never hardcoded at the call site.

    Degeneracy filter: any model where Viterbi hard-label predictions on
    training data assign fewer than min_regime_share of observations to any
    single regime is rejected.

    Fallback: if all seeds fail the filter, return the model with highest
    log-likelihood and emit a WARNING.

    Args:
        df_train:          In-sample feature DataFrame (DatetimeIndex required).
        n_regimes:         Number of HMM hidden states.
        n_init:            Number of seeds to try (set in regime_config.yaml).
        train_end_date:    IS boundary — forwarded to initialise_emissions().
        covariance_type:   HMM emission covariance type ('full' or 'diag').
        n_iter:            Max EM iterations per seed.
        tol:               EM convergence tolerance per seed.
        p_stay:            Diagonal element for initial transition matrix.
        min_regime_share:  Degeneracy filter: minimum fraction for any regime.
        feature_names:     Stored on the returned detector for interpretability.

    Returns:
        (best_detector, scaler): Best HMMRegimeDetector and its StandardScaler.

    Raises:
        RuntimeError: If all n_init seeds fail with exceptions.
    """
    # IS boundary is a dataset-level invariant — validate once before the loop
    # so the error propagates immediately rather than being silently swallowed
    # and misreported as "all seeds failed with exceptions".
    if train_end_date is not None and isinstance(df_train.index, pd.DatetimeIndex):
        train_max = df_train.index.max()
        if train_max > train_end_date:
            raise ValueError(
                "df_train contains out-of-sample dates: max date in df_train is %s "
                "but train_end_date is %s. Pass only in-sample data to "
                "fit_best_of_n_seeds()." % (train_max.date(), train_end_date.date())
            )

    X_raw = df_train.values
    candidates: list[dict] = []

    for seed in range(n_init):
        try:
            means, covs, scaler = initialise_emissions(
                df_train,
                n_clusters=n_regimes,
                random_state=seed,
                covariance_type=covariance_type,
                train_end_date=train_end_date,
            )
            X_scaled = scaler.transform(X_raw)

            detector = HMMRegimeDetector(
                n_regimes=n_regimes,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=seed,
                means=means,
                covars=covs,
                transmat=initialise_transitions(n_regimes, p_stay),
                startprob=initialise_probabilities(n_regimes),
                feature_names=feature_names,
            )
            detector.fit(X_scaled)

            ll = detector.score(X_scaled)
            labels = detector.predict(X_scaled)
            _, counts = np.unique(labels, return_counts=True)
            shares = counts / len(labels)
            passed = bool(np.min(shares) >= min_regime_share)

            logger.info(
                "fit_best_of_n_seeds: seed=%d  ll=%.4f  passed=%s  min_share=%.3f",
                seed, ll, passed, float(np.min(shares)),
            )
            candidates.append({
                "seed": seed,
                "detector": detector,
                "scaler": scaler,
                "ll": ll,
                "passed_filter": passed,
            })

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "fit_best_of_n_seeds: seed=%d raised %s: %s — skipping.",
                seed, type(exc).__name__, exc,
            )

    if not candidates:
        raise RuntimeError(
            f"fit_best_of_n_seeds: all {n_init} seeds failed with exceptions. "
            "Check input data and configuration."
        )

    passing = [c for c in candidates if c["passed_filter"]]
    if passing:
        best = max(passing, key=lambda c: c["ll"])
        logger.info(
            "fit_best_of_n_seeds: selected seed=%d (ll=%.4f) from %d/%d passing.",
            best["seed"], best["ll"], len(passing), len(candidates),
        )
    else:
        best = max(candidates, key=lambda c: c["ll"])
        logger.warning(
            "fit_best_of_n_seeds: no seed passed the degeneracy filter "
            "(min_regime_share=%.3f). Falling back to highest-LL seed=%d (ll=%.4f). "
            "Consider loosening min_regime_share in regime_config.yaml or inspecting features.",
            min_regime_share, best["seed"], best["ll"],
        )

    return best["detector"], best["scaler"]