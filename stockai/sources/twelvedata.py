"""Twelve Data: prețuri zilnice cu istoric lung. Plan gratuit: 800 cereri/zi, 8 pe minut."""

from __future__ import annotations

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

    def daily(self, symbol: str, outputsize: int = 5000) -> pd.DataFrame:
        self.limiter.wait()
        try:
            resp = self.session.get(
                f"{BASE_URL}/time_series",
                params={"symbol": symbol, "interval": "1day", "outputsize": outputsize, "apikey": self.api_key},
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise DataError(f"Twelve Data indisponibil pentru {symbol}: {exc}") from exc
        return parse_time_series(payload, symbol)


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
