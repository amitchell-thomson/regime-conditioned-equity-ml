import pandas as pd

def z_score(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return (df - df.rolling(window=window).mean()) / df.rolling(window=window).std()

