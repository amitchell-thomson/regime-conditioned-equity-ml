import pandas as pd
import numpy as np
from typing import Dict, Any

def evaluate_regime_stability(regimes: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate regime stability metrics.
    
    Returns:
        - persistence: Average regime duration (days)
        - n_transitions: Number of regime changes
        - entropy: Regime distribution entropy (balanced = high)
    """
    # Regime transitions
    transitions = np.diff(regimes)
    n_transitions = np.sum(transitions != 0)
    
    # Average persistence
    regime_lengths = []
    current_regime = regimes[0]
    current_length = 1
    
    for r in regimes[1:]:
        if r == current_regime:
            current_length += 1
        else:
            regime_lengths.append(current_length)
            current_regime = r
            current_length = 1
    regime_lengths.append(current_length)
    
    avg_persistence = np.mean(regime_lengths)
    
    # Regime distribution entropy
    unique, counts = np.unique(regimes, return_counts=True)
    probs = counts / len(regimes)
    entropy = -np.sum(probs * np.log2(probs))
    
    return {
        'avg_persistence': avg_persistence,
        'n_transitions': n_transitions,
        'regime_entropy': entropy,
        'regime_counts': dict(zip(unique, counts))
    }

def compare_models(
    features: pd.DataFrame,
    models: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compare multiple regime detection models.
    
    Args:
        features: Input features DataFrame (will be converted to numpy array)
        models: Dict of {name: model_instance}
        
    Returns:
        Comparison dataframe with BIC, AIC, log-likelihood
    """
    # Convert DataFrame to numpy array
    X = features.values
    
    results = []
    
    for name, model in models.items():
        model.fit(X)
        regimes = model.predict(X)
        
        # Get model metrics
        if hasattr(model, 'score'):
            ll = model.score(X)
            n_params = estimate_n_params(model)
            n_samples = len(X)
            
            bic = -2 * ll + n_params * np.log(n_samples)
            aic = -2 * ll + 2 * n_params
        else:
            ll, bic, aic = np.nan, np.nan, np.nan
        
        # Regime stability
        stability = evaluate_regime_stability(regimes)
        
        results.append({
            'model': name,
            'log_likelihood': ll,
            'bic': bic,
            'aic': aic,
            'avg_persistence': stability['avg_persistence'],
            'n_transitions': stability['n_transitions'],
            'entropy': stability['regime_entropy']
        })
    
    return pd.DataFrame(results).sort_values('bic')

def estimate_n_params(model) -> int:
    """Estimate number of parameters for BIC/AIC."""
    if hasattr(model, 'model'):  # HMM
        n_features = model.model.n_features
        n_states = model.model.n_components
        
        # Means, covariances, transition matrix, initial state
        if model.covariance_type == 'full':
            n_cov = n_states * n_features * (n_features + 1) // 2
        else:
            n_cov = n_states * n_features
        
        return n_states * n_features + n_cov + n_states**2 + n_states
    else:
        return 0  # Unknown