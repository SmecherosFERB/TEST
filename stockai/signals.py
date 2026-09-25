"""Semnale suplimentare, alese pentru că au dovezi publicate (vezi docs/surse-de-date.md).

Fiecare funcție `*_signal` întoarce un dicționar cu `score` în [-100, 100] (pozitiv = favorabil creșterii)
plus detaliile pe care le arătăm utilizatorului și lui Claude, sau None când nu avem date.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .indicators import sma


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


# ---------- serii din prețuri (folosite și de model) ----------

def momentum_12_1(close: pd.Series) -> pd.Series:
    """Randamentul pe ultimele 12 luni, fără ultima lună (Jegadeesh și Titman)."""
    return close.shift(21) / close.shift(252) - 1


def high_52w_gap(close: pd.Series) -> pd.Series:
    """Cât de departe e prețul de maximul ultimelor 52 de săptămâni (0 = chiar la maxim)."""
    return close / close.rolling(252, min_periods=200).max() - 1


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def regime_series(market_close: pd.Series) -> pd.Series:
    """Trendul pieței în [-1, 1]: prețul față de media pe 200 de zile și media pe 50 față de cea pe 200."""
    s200, s50 = sma(market_close, 200), sma(market_close, 50)
    out = 0.5 * np.sign(market_close - s200) + 0.5 * np.sign(s50 - s200)
    return out.where(s200.notna())


# ---------- semnale pentru ziua de azi ----------

def market_signal(market_close: pd.Series | None) -> dict[str, Any] | None:
    if market_close is None or len(market_close) < 220:
        return None
    regime = regime_series(market_close).iloc[-1]
    if regime != regime:
        return None
    ret_1m = float(market_close.iloc[-1] / market_close.iloc[-22] - 1)
    label = "în creștere" if regime > 0.5 else "în scădere" if regime < -0.5 else "fără direcție clară"
    return {
        "score": 100 * _clip(0.8 * regime + 0.2 * np.sign(ret_1m)),
        "label": label,
        "above_200d": bool(market_close.iloc[-1] > sma(market_close, 200).iloc[-1]),
        "return_1m": ret_1m,
    }


def macro_signal(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Stres pe piețe: VIX ridicat, prima de risc a obligațiunilor riscante în creștere, curba inversată."""
    if not snapshot:
        return None
    parts, notes = [], []
    vix = (snapshot.get("VIXCLS") or {}).get("value")
    if vix is not None:
        if vix >= 30:
            parts.append(-1.0)
            notes.append(f"VIX {vix:.1f}: frică mare pe piață")
        elif vix >= 22:
            parts.append(-0.4)
            notes.append(f"VIX {vix:.1f}: nervozitate peste medie")
        else:
            parts.append(0.2)
            notes.append(f"VIX {vix:.1f}: piață calmă")
    hy = snapshot.get("BAMLH0A0HYM2") or {}
    if hy.get("change_3m") is not None:
        ch = hy["change_3m"]
        if ch >= 0.75:
            parts.append(-1.0)
            notes.append(f"prima de risc pe obligațiuni a crescut cu {ch:.2f} pp în 3 luni: stres de credit")
        elif ch >= 0.3:
            parts.append(-0.5)
            notes.append(f"prima de risc pe obligațiuni în creștere (+{ch:.2f} pp în 3 luni)")
        elif ch <= -0.3:
            parts.append(0.3)
            notes.append(f"prima de risc pe obligațiuni în scădere ({ch:.2f} pp în 3 luni)")
        else:
            parts.append(0.0)
    curve = (snapshot.get("T10Y2Y") or {}).get("value")
    if curve is not None and curve < 0:
        parts.append(-0.3)
        notes.append(f"curba randamentelor inversată ({curve:.2f} pp): risc de recesiune")
    if not parts:
        return None
    return {"score": 100 * _clip(float(np.mean(parts))), "notes": notes}


def earnings_signal(quarters: list[dict[str, Any]] | None, as_of: date) -> dict[str, Any] | None:
    """Surpriza la ultimele rezultate (efectul se estompează în ~3 luni) și câte din ultimele 4 au depășit estimările."""
    if not quarters:
        return None
    past = [q for q in quarters if q["reported"] <= as_of]
    if not past:
        return None
    last = past[0]
    days = (as_of - last["reported"]).days
    last4 = [q["surprise_pct"] for q in past[:4] if q.get("surprise_pct") is not None]
    beats = sum(1 for s in last4 if s > 0)
    fade = 1.0 if days <= 60 else 0.5 if days <= 120 else 0.2
    surprise = last.get("surprise_pct")
    s_last = _clip(surprise / 10) if surprise is not None else 0.0
    s_beats = (beats - len(last4) / 2) / 2 if last4 else 0.0
    return {
        "score": 100 * _clip(0.7 * s_last * fade + 0.3 * s_beats),
        "last_reported": last["reported"].isoformat(),
        "days_since": days,
        "last_surprise_pct": surprise,
        "beats_last4": f"{beats}/{len(last4)}",
        "history": [
            {"reported": q["reported"].isoformat(), "eps": q.get("eps"), "estimate": q.get("estimate"), "surprise_pct": q.get("surprise_pct")}
            for q in past[:4]
        ],
    }


def earnings_surprise_series(quarters: list[dict[str, Any]] | None, index: pd.DatetimeIndex, window_days: int = 60) -> pd.Series:
    """Pentru model: surpriza ultimului raport, doar în cele `window_days` zile de după publicare (altfel 0)."""
    out = pd.Series(0.0, index=index)
    for q in sorted(quarters or [], key=lambda q: q["reported"]):
        s = q.get("surprise_pct")
        if s is None:
            continue
        start = pd.Timestamp(q["reported"]) + pd.Timedelta(days=1)  # a doua zi, ca să nu folosim informație din viitor
        end = start + pd.Timedelta(days=window_days)
        out[(out.index >= start) & (out.index < end)] = _clip(s / 100, -0.5, 0.5)
    return out


def revision_signal(trend: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reviziile estimărilor analiștilor prezic randamentele (Chan, Jegadeesh și Lakonishok, 1996).
    O revizie medie în sus de 3% (jumătate pe 30, jumătate pe 90 de zile) = +100."""
    if not trend:
        return None
    parts = [0.5 * (r["current"] / r["30_days_ago"] - 1) + 0.5 * (r["current"] / r["90_days_ago"] - 1) for r in trend.values()]
    if not parts:
        return None
    change = float(np.mean(parts))
    return {"score": 100 * _clip(change / 0.03), "change": round(change, 4), "estimates": trend}


def insider_signal(trades: list[dict[str, Any]] | None, as_of: date, days: int = 180) -> dict[str, Any] | None:
    """Cumpărările pe piață ale mai multor insideri sunt semnalul informativ; vânzările sunt adesea de rutină."""
    if trades is None:
        return None
    since = as_of - timedelta(days=days)
    recent = [t for t in trades if since <= t["date"] <= as_of]
    buys = [t for t in recent if t["code"] == "P"]
    sells = [t for t in recent if t["code"] == "S"]
    buyers = {t["owner"] for t in buys}
    sellers = {t["owner"] for t in sells}
    score = 80 if len(buyers) >= 2 else 40 if buyers else (-20 if len(sellers) >= 3 else 0)
    return {
        "score": float(score),
        "days": days,
        "buys": len(buys),
        "buyers": len(buyers),
        "buy_value": round(sum(t["shares"] * t["price"] for t in buys)),
        "sells": len(sells),
        "sellers": len(sellers),
        "sell_value": round(sum(t["shares"] * t["price"] for t in sells)),
    }
