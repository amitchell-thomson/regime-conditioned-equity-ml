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

def get_top_features(n: int = 15) -> List[str]:
    """Return recommended top N features for regime modeling."""
    # Priority order based on regime relevance
    top = [
        'T10Y3M_level_zscore_252',           # #1: Curve slope (recession)
        'VIXCLS_level_zscore_63',            # #2: Stress
        'NFCI_level_zscore_50',              # #3: Credit conditions
        'PCEPILFE_yoy_12_zscore_36',         # #4: Inflation
        'DGS10_level_zscore_252',            # #5: Rate level
        'CFNAI_level_zscore_36',             # #6: Growth
        'T10Y3M_diff_21_zscore_252',         # #7: Curve dynamics
        'VIXCLS_diff_5_zscore_126',          # #8: Vol momentum
        'PCEPILFE_yoy_12_diff_1_zscore_36',  # #9: Inflation accel
        'DGS10_diff_21_zscore_252',          # #10: Rate momentum
        'NFCI_diff_21_zscore_50',            # #11: Credit tightening
        'CFNAI_diff_1_zscore_36',            # #12: Growth momentum
    ]
    return top[:n]