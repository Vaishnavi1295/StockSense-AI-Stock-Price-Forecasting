from pathlib import Path
import numpy as np
import pandas as pd

def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()

def load_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download historical data. Raises a helpful error if retrieval fails."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is missing. Run: pip install -r requirements.txt") from exc
    symbol = normalize_ticker(ticker)
    if not symbol:
        raise ValueError("Enter a stock ticker, for example RELIANCE.NS or AAPL.")
    raw = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"No historical data was returned for '{symbol}'. Check the ticker or try another period.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).title().replace(" ", "") for c in raw.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"The data provider did not return expected columns: {missing}")
    data = raw[required].copy()
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Close"])
    data["Volume"] = data["Volume"].fillna(0)
    data = data[data["Close"] > 0]
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} valid rows were received. Try a longer history or another ticker.")
    return data

def load_demo_data() -> pd.DataFrame:
    """Create deterministic synthetic data for offline UI testing. Not real stock data."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=750)
    returns = rng.normal(0.00035, 0.014, len(dates))
    close = 150 * np.exp(np.cumsum(returns))
    open_price = close * (1 + rng.normal(0, 0.004, len(dates)))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.012, len(dates)))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.012, len(dates)))
    volume = rng.integers(500_000, 5_000_000, len(dates))
    return pd.DataFrame({"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
