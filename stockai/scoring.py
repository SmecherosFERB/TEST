"""Transformă indicatorii, datele fundamentale și știrile în scoruri de la -100 la +100.

Convenție: pozitiv = semnal de creștere (BUY), negativ = semnal de scădere (SELL).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TECHNICAL_WEIGHTS = {"trend": 0.35, "macd": 0.25, "rsi": 0.2, "bollinger": 0.1, "volume": 0.1}


def technical_components(ind: pd.DataFrame) -> pd.DataFrame:
    """Fiecare componentă tehnică în [-1, 1], calculată pentru fiecare zi."""
    trend = 0.5 * np.sign(ind["close"] - ind["sma50"]) + 0.5 * np.sign(ind["sma50"] - ind["sma200"])
    macd_part = 0.5 * np.sign(ind["macd_hist"]) + 0.5 * np.sign(ind["macd_hist"].diff())
    # RSI mic = supravândut = șansă de revenire; RSI mare = supracumpărat.
    rsi_part = ((50 - ind["rsi"]) / 20).clip(-1, 1)
    boll_part = ((0.5 - ind["pct_b"]) * 2).clip(-1, 1)
    # Volum neobișnuit de mare confirmă direcția zilei.
    volume_part = np.sign(ind["daily_return"]).where(ind["volume_ratio"] > 1.5, 0.0)
    comps = pd.DataFrame(
        {
            "trend": trend,
            "macd": macd_part,
            "rsi": rsi_part,
            "bollinger": boll_part,
            "volume": volume_part,
        }
    )
    # Zilele fără istoric suficient (ex. înainte de SMA200) rămân NaN.
    return comps.where(ind[["sma200", "rsi", "pct_b", "volume_ratio"]].notna().all(axis=1))


def technical_score(ind: pd.DataFrame) -> pd.Series:
    comps = technical_components(ind)
    weights = pd.Series(TECHNICAL_WEIGHTS)
    return 100 * comps[weights.index].mul(weights).sum(axis=1, min_count=len(weights))


def fundamental_score(info: dict[str, Any]) -> float | None:
    """Media semnalelor fundamentale disponibile; None dacă nu avem niciunul."""
    signals: list[float] = []

    trailing_pe, forward_pe = info.get("trailingPE"), info.get("forwardPE")
    if trailing_pe and forward_pe and trailing_pe > 0 and forward_pe > 0:
        # P/E viitor mai mic decât cel actual = analiștii așteaptă creștere de profit.
        signals.append(float(np.clip((trailing_pe - forward_pe) / trailing_pe * 4, -1, 1)))

    growth = info.get("revenueGrowth")
    if growth is not None:
        signals.append(float(np.clip(growth / 0.15, -1, 1)))

    margin = info.get("profitMargins")
    if margin is not None:
        signals.append(float(np.clip(margin / 0.15, -1, 1)))

    debt = info.get("debtToEquity")
    if debt is not None:
        # Sub 50% datorii/capital = sănătos, peste 200% = riscant.
        signals.append(float(np.clip((125 - debt) / 75, -1, 1)))

    rec = info.get("recommendationMean")
    if rec is not None:
        # 1 = Strong Buy ... 5 = Sell la analiști.
        signals.append(float(np.clip((3 - rec) / 2, -1, 1)))

    return 100 * float(np.mean(signals)) if signals else None


def sentiment_score(news: list[dict[str, Any]]) -> float | None:
    """Media sentimentului din știri, ponderată cu relevanța (scala Alpha Vantage)."""
    scored = [n for n in news if n.get("sentiment") is not None]
    if not scored:
        return None
    weights = np.array([n.get("relevance") or 1.0 for n in scored], dtype=float)
    values = np.array([n["sentiment"] for n in scored], dtype=float)
    mean = float(np.average(values, weights=weights)) if weights.sum() > 0 else float(values.mean())
    # Alpha Vantage consideră ±0.35 deja „bullish/bearish”; scalăm ca acolo să fie ±100.
    return 100 * float(np.clip(mean / 0.35, -1, 1))


def composite_score(components: dict[str, float | None], weights: dict[str, float]) -> float:
    """Media ponderată a componentelor disponibile (ponderile se renormalizează)."""
    available = {k: v for k, v in components.items() if v is not None and k in weights}
    if not available:
        return 0.0
    total = sum(weights[k] for k in available)
    return sum(weights[k] * v for k, v in available.items()) / total


def decision_from_score(score: float, threshold: float) -> str:
    if score >= threshold:
        return "BUY"
    if score <= -threshold:
        return "SELL"
    return "HOLD"
