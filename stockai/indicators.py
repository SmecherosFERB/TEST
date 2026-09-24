"""Indicatori tehnici calculați direct cu pandas (fără dependențe externe)."""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI Wilder: 0-100; sub 30 = supravândut, peste 70 = supracumpărat."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    # Fără nicio scădere în fereastră, RSI este 100 prin definiție.
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger_pct_b(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Poziția prețului în benzile Bollinger: 0 = banda de jos, 1 = banda de sus."""
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper, lower = mid + num_std * std, mid - num_std * std
    return (close - lower) / (upper - lower)


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volumul de azi raportat la media ultimelor `window` zile."""
    return volume / volume.rolling(window).mean()


def compute_all(prices: pd.DataFrame) -> pd.DataFrame:
    """Primește OHLCV (coloane Close, Volume) și întoarce toți indicatorii pe zi."""
    close, volume = prices["Close"], prices["Volume"]
    m = macd(close)
    return pd.DataFrame(
        {
            "close": close,
            "sma50": sma(close, 50),
            "sma200": sma(close, 200),
            "rsi": rsi(close),
            "macd": m["macd"],
            "macd_signal": m["signal"],
            "macd_hist": m["hist"],
            "pct_b": bollinger_pct_b(close),
            "volume_ratio": volume_ratio(volume),
            "daily_return": close.pct_change(),
        }
    )
