# src/regime_ml/features/selection.py

import pandas as pd
from typing import List, Optional
from regime_ml.data.common.loaders import load_dataframe

def get_feature_groups() -> dict:
    """Return features grouped by category."""
    return {
        'stress': ['VIXCLS_level_zscore_63', 'VIXCLS_diff_5_zscore_126', 
                   'NFCI_level_zscore_50', 'NFCI_diff_21_zscore_50'],
        'rates': ['DGS2_level_zscore_252', 'DGS10_level_zscore_252', 
                  'T10Y3M_level_zscore_252', 'T10Y3M_diff_21_zscore_252'],
        'inflation': ['PCEPILFE_yoy_12_zscore_36', 'PCEPILFE_yoy_12_diff_1_zscore_36'],
        'growth': ['CFNAI_level_zscore_36', 'INDPRO_yoy_12_zscore_36'],
        'labor': ['ICSA_movingaverage_4_zscore_50', 'ICSA_ma_4_pctchange_13_zscore_50']
    }

def get_top_features(n: int = 5) -> list[str]:
    """
    Return the top N most important features for regime detection.
    
    Features are ranked by regime discriminative power based on:
    - Economic theory (recession signals, stress indicators)
    - Empirical importance in regime literature
    - Balance of levels (states) vs momentum (transitions)
    
    Args:
        n: Number of top features to return (default: 5)
        
    Returns:
        List of feature names, ordered by importance
    """
    
    # Ranked features with concise explanations
    ranked_features = [
        # Tier 1: Core regime indicators (THE big 5)
        'T10Y3M_level_zscore_252',                  # 1. Curve slope - recession predictor, inversion signals downturn
        'VIXCLS_level_zscore_63',                   # 2. Equity vol - risk-on vs risk-off, fear gauge
        'NFCI_level_zscore_50',                     # 3. Financial stress - credit conditions, systemic risk
        'PCEPILFE_yoy_12_zscore_36',                # 4. Inflation - policy regime (dovish/hawkish), stagflation risk
        'CFNAI_level_zscore_36',                    # 5. Activity - expansion vs contraction, business cycle
        
        # Tier 2: Critical momentum & secondary levels (useful for 6-10 feature models)
        'DGS10_level_zscore_252',                   # 6. Long rates - QE/QT regime, growth expectations
        'VIXCLS_diff_5_zscore_126',                 # 7. Vol momentum - stress building/easing rapidly
        'T10Y3M_diff_21_zscore_252',                # 8. Curve steepening - reflation vs flattening trades
        'CFNAI_diff_1_zscore_36',                   # 9. Growth momentum - acceleration/deceleration
        'NFCI_diff_21_zscore_50',                   # 10. Credit tightening - financial conditions changing
        
        # Tier 3: Useful complementary features (for 11-15 feature models)
        'PCEPILFE_yoy_12_diff_1_zscore_36',         # 11. Inflation accel - disinflation vs reacceleration
        'DGS10_diff_21_zscore_252',                 # 12. Rate momentum - Fed tightening/easing pace
        'ICSA_movingaverage_4_pctchange_13_zscore_50',  # 13. Labor deterioration - unemployment rising (3mo)
        'INDPRO_yoy_12_zscore_36',                  # 14. Production - manufacturing cycle, goods vs services
        'DGS2_level_zscore_252',                    # 15. Short rates - immediate policy stance
        
        # Tier 4: Additional depth (for 16-20 feature models)
        'T10Y3M_diff_5_zscore_126',                 # 16. Curve momentum - quick flattening signals
        'VIXCLS_movingaverage_21_diff_21_zscore_252',  # 17. Vol trend - sustained stress vs spikes
        'INDPRO_yoy_12_diff_1_zscore_36',           # 18. Production accel - capex cycle turning points
        'DGS10_diff_1_rollingstd_21_zscore_126',    # 19. Rate volatility - policy uncertainty
        'ICSA_movingaverage_4_zscore_50',           # 20. Claims level - labor market tightness
        
        # Tier 5: Marginal value (for 21-25 feature models, consider dropping)
        'NFCI_diff_5_diff_5_zscore_25',             # 21. Credit accel - second derivative, noisy
        'VIXCLS_pctchange_1_rollingstd_21_zscore_126',  # 22. Vol-of-vol - extreme stress indicator
        'ICSA_movingaverage_4_pctchange_4_zscore_25',  # 23. Labor momentum - 1mo change, noisier
        'PCEPILFE_pctchange_1_zscore_36',           # 24. Monthly inflation - too noisy for regimes
        'INDPRO_pctchange_3_zscore_36',             # 25. 3mo production - overlaps with YoY and CFNAI
    ]
    
    # Return top N features
    return ranked_features[:n]