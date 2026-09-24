"""Alpha Vantage (gratuit, 25 cereri/zi): rezultate trimestriale, insideri, știri cu sentiment."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests

from ..cache import RateLimiter
from ..errors import DataError

BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantage:
    def __init__(self, api_key: str, session: requests.Session | None = None, min_interval: float = 1.3) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()
        self.limiter = RateLimiter(min_interval)

    def _get(self, function: str, **params: Any) -> dict[str, Any]:
        self.limiter.wait()
        try:
            resp = self.session.get(BASE_URL, params={"function": function, "apikey": self.api_key, **params}, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise DataError(f"Alpha Vantage indisponibil ({function}): {exc}") from exc
        problem = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
        if problem:
            raise DataError(f"Alpha Vantage ({function}): {problem}")
        return payload

    def earnings(self, symbol: str) -> list[dict[str, Any]]:
        return parse_earnings(self._get("EARNINGS", symbol=symbol))

    def insiders(self, symbol: str) -> list[dict[str, Any]]:
        return parse_insiders(self._get("INSIDER_TRANSACTIONS", symbol=symbol))

    def news(self, symbol: str, limit: int = 50) -> list[dict[str, Any]]:
        from ..data import parse_alpha_vantage_feed

        payload = self._get("NEWS_SENTIMENT", tickers=symbol, limit=limit)
        return parse_alpha_vantage_feed(payload.get("feed", []), symbol)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def parse_earnings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Trimestrele raportate, cele mai recente primele."""
    out = []
    for q in payload.get("quarterlyEarnings", []):
        reported = _date(q.get("reportedDate"))
        if reported is None:
            continue
        out.append(
            {
                "reported": reported,
                "fiscal_end": q.get("fiscalDateEnding"),
                "eps": _num(q.get("reportedEPS")),
                "estimate": _num(q.get("estimatedEPS")),
                "surprise_pct": _num(q.get("surprisePercentage")),
            }
        )
    return sorted(out, key=lambda q: q["reported"], reverse=True)


def parse_insiders(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Tranzacții brute. Alpha Vantage nu dă codul tranzacției (cumpărare pe piață vs. acordare)."""
    out = []
    for t in payload.get("data", []):
        when = _date(t.get("transaction_date"))
        if when is None:
            continue
        out.append(
            {
                "date": when,
                "owner": t.get("executive") or "",
                "title": t.get("executive_title") or "",
                "security": t.get("security_type") or "",
                "side": t.get("acquisition_or_disposal") or "",
                "shares": _num(t.get("shares")) or 0.0,
                "price": _num(t.get("share_price")),
            }
        )
    return out


def classify_insiders(rows: list[dict[str, Any]], close: pd.Series, tolerance: float = 0.15) -> list[dict[str, Any]]:
    """Aproximează cumpărările/vânzările pe piață: acțiuni comune, preț apropiat de cel al bursei în ziua respectivă.

    Acordările (preț 0) și exercitările de opțiuni (preț mult sub piață) sunt excluse.
    """
    if close.empty:
        return []
    out = []
    for r in rows:
        price = r.get("price")
        if not price or price <= 0 or "common" not in r["security"].lower():
            continue
        market = close.asof(pd.Timestamp(r["date"]))
        if market != market or market <= 0 or abs(price / market - 1) > tolerance:
            continue
        code = "P" if r["side"] == "A" else "S" if r["side"] == "D" else None
        if code:
            out.append({"date": r["date"], "owner": r["owner"], "title": r["title"], "code": code,
                        "shares": r["shares"], "price": price})
    return out
