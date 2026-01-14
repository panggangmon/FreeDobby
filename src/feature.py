import numpy as np
import pandas as pd


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    # Simple RSI using rolling mean of gains/losses
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(window=window).mean()
    return atr / close.replace(0, np.nan)


def _zscore(s: pd.Series, window: int = 20) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    z = (s - m) / sd.replace(0, np.nan)
    return z


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared feature builder for training and live inference.
    Expects columns: open, high, low, close, volume.
    """
    close = df["close"]

    # Trend / overbought
    df["ma_5"] = _sma(close, 5)
    df["ma_20"] = _sma(close, 20)
    df["ma_60"] = _sma(close, 60)
    df["rsi_14"] = _rsi(close, 14)

    # Returns
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ret_20"] = close.pct_change(20)

    # Volatility / risk proxy
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["atr_14_pct"] = _atr_pct(df, 14)

    # Volume anomaly
    df["dollar_volume"] = df["close"] * df["volume"]
    df["vol_z_20"] = _zscore(df["volume"], 20)

    # Trend strength
    df["dist_ma20"] = (df["close"] / df["ma_20"]) - 1.0
    df["ma20_slope_5"] = df["ma_20"].pct_change(5)

    return df
