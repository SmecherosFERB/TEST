from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_prices(n: int = 1260, drift: float = 0.0004, vol: float = 0.015, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(returns))
    index = pd.bdate_range(end="2026-09-23", periods=n)
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": volume},
        index=index,
    )


class FakeData:
    def __init__(self, prices: pd.DataFrame, fundamentals: dict | None = None, news: list | None = None):
        self._prices = prices
        self._fundamentals = fundamentals or {}
        self._news = news or []

    def prices(self, ticker, period):
        return self._prices

    def fundamentals(self, ticker):
        return self._fundamentals

    def news(self, ticker):
        return self._news


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()
