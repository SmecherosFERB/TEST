"""FRED (Fed St. Louis, gratuit): context macro. Cu cheie folosește API-ul, fără cheie fișierul CSV public."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import requests

from ..cache import DiskCache, RateLimiter
from ..errors import DataError

API_URL = "https://api.stlouisfed.org/fred/series/observations"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Seriile folosite: volatilitatea așteptată, curba randamentelor, prima de risc la obligațiuni riscante, dobânda la 10 ani.
MACRO_SERIES = {
    "VIXCLS": "indicele de volatilitate VIX",
    "T10Y2Y": "diferența dobânzi 10 ani - 2 ani (curba randamentelor)",
    "BAMLH0A0HYM2": "prima de risc la obligațiuni high-yield (puncte procentuale)",
    "DGS10": "dobânda la obligațiunile SUA pe 10 ani",
}


class Fred:
    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()
        self.cache = cache or DiskCache()
        self.limiter = RateLimiter(0.6)

    def series(self, series_id: str, start: str = "2000-01-01") -> pd.Series:
        cached = self.cache.get_frame(f"fred_{series_id}", max_age_hours=12)
        if cached is not None and not cached.empty:
            return cached.iloc[:, 0].dropna()
        self.limiter.wait()
        try:
            if self.api_key:
                resp = self.session.get(
                    API_URL,
                    params={"series_id": series_id, "api_key": self.api_key, "file_type": "json", "observation_start": start},
                    timeout=30,
                )
                resp.raise_for_status()
                data = parse_fred_json(resp.json())
            else:
                resp = self.session.get(CSV_URL, params={"id": series_id, "cosd": start}, timeout=30)
                resp.raise_for_status()
                data = parse_fred_csv(resp.text)
        except (requests.RequestException, ValueError) as exc:
            raise DataError(f"FRED indisponibil pentru {series_id}: {exc}") from exc
        self.cache.set_frame(f"fred_{series_id}", data.to_frame(series_id))
        return data

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for sid in MACRO_SERIES:
            try:
                s = self.series(sid)
            except DataError:
                continue
            if s.empty:
                continue
            out[sid] = {
                "value": float(s.iloc[-1]),
                "change_3m": float(s.iloc[-1] - s.iloc[-64]) if len(s) > 64 else None,
                "as_of": str(s.index[-1].date()),
            }
        return out


def parse_fred_json(payload: dict[str, Any]) -> pd.Series:
    rows = [(o.get("date"), o.get("value")) for o in payload.get("observations", [])]
    s = pd.Series({pd.Timestamp(d): v for d, v in rows if d})
    return pd.to_numeric(s, errors="coerce").dropna().sort_index()


def parse_fred_csv(text: str) -> pd.Series:
    frame = pd.read_csv(io.StringIO(text))
    if frame.shape[1] < 2:
        raise ValueError("CSV FRED neașteptat")
    # Prima coloană e data (antet „DATE” sau „observation_date”), a doua valoarea; lipsurile sunt „.”.
    s = pd.Series(frame.iloc[:, 1].values, index=pd.to_datetime(frame.iloc[:, 0]))
    return pd.to_numeric(s, errors="coerce").dropna().sort_index()
