import pandas as pd

def moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return pd.DataFrame(df.rolling(window=window).mean())