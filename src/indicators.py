import numpy as np
import pandas as pd

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["Close"].astype(float)
    result["SMA_20"] = close.rolling(20, min_periods=20).mean()
    result["SMA_50"] = close.rolling(50, min_periods=50).mean()
    result["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    result["RSI_14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
    return result
