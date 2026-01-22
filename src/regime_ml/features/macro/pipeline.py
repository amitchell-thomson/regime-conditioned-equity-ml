import pandas as pd
from regime_ml.features.common.transforms import (
    BaseTransform, 
    ChainedTransform
)

def run_macro_feature_pipeline() -> pd.DataFrame:
    """
    Run the complete macro feature pipeline from processed data to features.
    """
    return pd.DataFrame()