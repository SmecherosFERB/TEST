import numpy as np
import pandas as pd

from stockai.indicators import bollinger_pct_b, compute_all, rsi, sma


def test_sma_matches_manual_mean():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(close, 3).iloc[-1] == 4.0


def test_rsi_is_100_when_price_only_rises():
    close = pd.Series(np.arange(1, 40, dtype=float))
    assert rsi(close).iloc[-1] == 100.0


def test_rsi_stays_in_range(prices):
    values = rsi(prices["Close"]).dropna()
    assert values.between(0, 100).all()


def test_pct_b_above_one_after_price_jump():
    close = pd.Series([100.0] * 19 + [101.0, 99.0] * 5 + [130.0])
    assert bollinger_pct_b(close).iloc[-1] > 1


def test_compute_all_has_values_after_warmup(prices):
    ind = compute_all(prices)
    assert ind.iloc[-1].notna().all()
    assert ind["sma200"].iloc[:199].isna().all()
