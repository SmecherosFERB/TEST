"""Twelve Data: prețuri zilnice cu istoric lung, fundamente, rezultate, insideri. Plan gratuit: 800 cereri/zi, 8 pe minut."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd
import requests

from ..cache import RateLimiter
from ..errors import DataError

BASE_URL = "https://api.twelvedata.com"


class TwelveData:
    def __init__(self, api_key: str, session: requests.Session | None = None, min_interval: float = 7.6) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()
        self.limiter = RateLimiter(min_interval)

    def _get(self, endpoint: str, symbol: str, **params: Any) -> dict[str, Any]:
        self.limiter.wait()
        try:
            resp = self.session.get(
                f"{BASE_URL}/{endpoint}", params={"symbol": symbol, "apikey": self.api_key, **params}, timeout=30
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise DataError(f"Twelve Data indisponibil pentru {symbol}: {exc}") from exc
        if payload.get("status") == "error":
            raise DataError(f"Twelve Data, {symbol}: {payload.get('message') or 'eroare'}")
        return payload

    def daily(self, symbol: str, outputsize: int = 5000) -> pd.DataFrame:
        return parse_time_series(self._get("time_series", symbol, interval="1day", outputsize=outputsize), symbol)

    def statistics(self, symbol: str) -> dict[str, Any]:
        return parse_statistics(self._get("statistics", symbol))

    def earnings(self, symbol: str) -> list[dict[str, Any]]:
        return self.earnings_report(symbol)["quarters"]

    def earnings_report(self, symbol: str) -> dict[str, Any]:
        """Trimestrele raportate și data următorului raport, dintr-o singură cerere."""
        payload = self._get("earnings", symbol)
        return {"quarters": parse_earnings(payload), "next": parse_next_earnings(payload)}

    def insiders(self, symbol: str) -> list[dict[str, Any]]:
        return parse_insiders(self._get("insider_transactions", symbol))


def parse_time_series(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
    if payload.get("status") == "error" or "values" not in payload:
        raise DataError(f"Twelve Data, {symbol}: {payload.get('message') or 'răspuns fără prețuri'}")
    frame = pd.DataFrame(payload["values"])
    if frame.empty:
        raise DataError(f"Twelve Data nu are prețuri pentru {symbol}")
    frame.index = pd.to_datetime(frame.pop("datetime"))
    if "volume" not in frame:
        frame["volume"] = 0
    frame = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    frame.columns = ["Open", "High", "Low", "Close", "Volume"]
    return frame.dropna(subset=["Close"]).sort_index()


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def parse_statistics(payload: dict[str, Any]) -> dict[str, Any]:
    """Aceleași chei ca la Yahoo Finance, ca scorul fundamental să funcționeze la fel."""
    st = payload.get("statistics") or {}
    val = st.get("valuations_metrics") or {}
    fin = st.get("financials") or {}
    inc = fin.get("income_statement") or {}
    bal = fin.get("balance_sheet") or {}
    stock = st.get("stock_statistics") or {}
    px = st.get("stock_price_summary") or {}
    splits = st.get("dividends_and_splits") or {}
    out = {
        "shortName": (payload.get("meta") or {}).get("name"),
        "marketCap": _num(val.get("market_capitalization")),
        "trailingPE": _num(val.get("trailing_pe")),
        "forwardPE": _num(val.get("forward_pe")),
        "revenueGrowth": _num(inc.get("quarterly_revenue_growth")),
        "earningsGrowth": _num(inc.get("quarterly_earnings_growth_yoy")),
        "profitMargins": _num(fin.get("profit_margin")),
        "debtToEquity": _num(bal.get("total_debt_to_equity_mrq")),
        "shortPercent": _num(stock.get("short_percent_of_shares_outstanding")),
        "shortRatio": _num(stock.get("short_ratio")),
        "fiftyTwoWeekHigh": _num(px.get("fifty_two_week_high")),
        "fiftyTwoWeekLow": _num(px.get("fifty_two_week_low")),
        "twoHundredDayAverage": _num(px.get("day_200_ma")),
        "lastSplitFactor": splits.get("last_split_factor") or None,
        "lastSplitDate": str(splits["last_split_date"])[:10] if splits.get("last_split_date") else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def parse_earnings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Trimestrele deja raportate (cu EPS efectiv), cele mai recente primele."""
    out = []
    for q in payload.get("earnings") or []:
        eps = _num(q.get("eps_actual"))
        try:
            reported = date.fromisoformat(str(q.get("date", ""))[:10])
        except ValueError:
            continue
        if eps is None or reported > date.today():
            continue
        out.append({"reported": reported, "fiscal_end": None, "eps": eps,
                    "estimate": _num(q.get("eps_estimate")), "surprise_pct": _num(q.get("surprise_prc"))})
    return sorted(out, key=lambda q: q["reported"], reverse=True)


def parse_next_earnings(payload: dict[str, Any], today: date | None = None) -> date | None:
    """Data următorului raport: primul rând de azi încolo care nu are încă EPS efectiv."""
    today = today or date.today()
    upcoming = []
    for q in payload.get("earnings") or []:
        if _num(q.get("eps_actual")) is not None:
            continue
        try:
            d = date.fromisoformat(str(q.get("date", ""))[:10])
        except ValueError:
            continue
        if d >= today:
            upcoming.append(d)
    return min(upcoming) if upcoming else None


_PRICE = re.compile(r"price\s+([\d.,]+)", re.IGNORECASE)


def parse_insiders(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Cumpărări (P) și vânzări (S) pe piață; acordările și cadourile (fără „Purchase”/„Sale”) sunt ignorate."""
    out = []
    for t in payload.get("insider_transactions") or []:
        desc = str(t.get("description") or "")
        low = desc.lstrip().lower()
        code = "P" if low.startswith(("purchase", "buy")) else "S" if low.startswith(("sale", "sell")) else None
        if code is None:
            continue
        try:
            when = date.fromisoformat(str(t.get("date_reported", ""))[:10])
        except ValueError:
            continue
        shares = _num(t.get("shares")) or 0.0
        m = _PRICE.search(desc)
        price = _num(m.group(1).replace(",", "")) if m else None
        if price is None and shares and _num(t.get("value")) is not None:
            price = _num(t.get("value")) / shares
        if price is None:
            continue
        out.append({"date": when, "owner": t.get("full_name") or "", "title": t.get("position") or "",
                    "code": code, "shares": shares, "price": price})
    return out
