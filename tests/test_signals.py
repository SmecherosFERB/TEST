from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stockai.signals import (
    earnings_signal,
    earnings_surprise_series,
    high_52w_gap,
    insider_signal,
    macro_signal,
    market_signal,
    momentum_12_1,
)

from .conftest import make_prices


def test_momentum_skips_last_month():
    close = pd.Series(np.arange(1, 301, dtype=float))
    # la ultima zi: prețul de acum 21 de zile / prețul de acum 252 de zile - 1
    assert momentum_12_1(close).iloc[-1] == pytest.approx(close.iloc[-22] / close.iloc[-253] - 1)


def test_high_52w_gap_is_zero_at_new_high():
    close = pd.Series(np.linspace(10, 20, 260))
    assert high_52w_gap(close).iloc[-1] == pytest.approx(0.0)


def test_market_signal_follows_trend():
    up = make_prices(drift=0.002, vol=0.005)["Close"]
    down = make_prices(drift=-0.002, vol=0.005)["Close"]
    assert market_signal(up)["score"] > 50 and market_signal(up)["label"] == "în creștere"
    assert market_signal(down)["score"] < -50
    assert market_signal(up.iloc[:100]) is None


def test_macro_signal_flags_stress():
    calm = macro_signal({"VIXCLS": {"value": 14.0}, "BAMLH0A0HYM2": {"value": 3.0, "change_3m": -0.4}})
    stress = macro_signal({"VIXCLS": {"value": 35.0}, "BAMLH0A0HYM2": {"value": 6.0, "change_3m": 1.2},
                           "T10Y2Y": {"value": -0.4}})
    assert calm["score"] > 0 and stress["score"] < -50
    assert any("VIX" in n for n in stress["notes"])
    assert macro_signal({}) is None


QUARTERS = [
    {"reported": date(2026, 7, 30), "eps": 2.02, "estimate": 1.88, "surprise_pct": 7.4},
    {"reported": date(2026, 4, 30), "eps": 2.01, "estimate": 1.94, "surprise_pct": 3.6},
    {"reported": date(2026, 1, 29), "eps": 2.84, "estimate": 2.67, "surprise_pct": 6.4},
    {"reported": date(2025, 10, 30), "eps": 1.85, "estimate": 1.77, "surprise_pct": -1.0},
]


def test_earnings_signal_fades_with_time():
    fresh = earnings_signal(QUARTERS, date(2026, 8, 20))
    old = earnings_signal(QUARTERS, date(2026, 12, 20))
    assert fresh["beats_last4"] == "3/4" and fresh["days_since"] == 21
    assert fresh["score"] > old["score"] > 0
    # nu folosim rapoarte publicate după data analizei
    assert earnings_signal(QUARTERS, date(2025, 11, 1))["last_reported"] == "2025-10-30"
    assert earnings_signal(QUARTERS, date(2025, 1, 1)) is None


def test_earnings_surprise_series_has_no_lookahead():
    idx = pd.date_range("2026-07-28", periods=80, freq="D")
    s = earnings_surprise_series(QUARTERS, idx)
    assert s[pd.Timestamp("2026-07-30")] == 0.0  # ziua raportului: încă necunoscut înainte de închidere
    assert s[pd.Timestamp("2026-07-31")] == pytest.approx(0.074)
    assert s[pd.Timestamp("2026-10-10")] == 0.0  # după 60 de zile efectul e considerat consumat


def test_insider_signal_rewards_cluster_buying():
    today = date(2026, 9, 24)
    t = lambda owner, code, days: {"date": today - timedelta(days=days), "owner": owner, "code": code, "shares": 100, "price": 50.0}
    cluster = insider_signal([t("A", "P", 10), t("B", "P", 20), t("C", "S", 5)], today)
    assert cluster["score"] == 80 and cluster["buyers"] == 2 and cluster["buy_value"] == 10000
    sellers = insider_signal([t("A", "S", 10), t("B", "S", 10), t("C", "S", 10), t("D", "P", 400)], today)
    assert sellers["score"] == -20 and sellers["buys"] == 0
    assert insider_signal([], today)["score"] == 0
    assert insider_signal(None, today) is None
