"""Surse de date. Toate sunt gratuite; fiecare sursă fără cheie sau fără răspuns e pur și simplu omisă.

- prețuri zilnice: Twelve Data (dacă ai TWELVE_DATA_API_KEY), altfel Yahoo Finance (neoficial);
- fundamentale: Twelve Data, altfel Yahoo Finance;
- știri cu sentiment: Alpha Vantage (ALPHA_VANTAGE_API_KEY);
- rezultate trimestriale: Twelve Data, altfel Alpha Vantage;
- tranzacțiile insiderilor: SEC EDGAR (SEC_USER_AGENT), altfel Twelve Data, altfel Alpha Vantage;
- context macro: FRED (FRED_API_KEY opțional);
- trendul pieței: indicele S&P 500 prin ETF-ul SPY.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Protocol

import pandas as pd

from .cache import DiskCache
from .errors import DataError
from .quality import drop_partial_bar

log = logging.getLogger(__name__)

FUNDAMENTAL_KEYS = (
    "shortName",
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "revenueGrowth",
    "earningsGrowth",
    "profitMargins",
    "debtToEquity",
    "recommendationMean",
    "recommendationKey",
    "targetMeanPrice",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
)

MARKET_SYMBOL = "SPY"


class DataProvider(Protocol):
    """Minimul necesar. Metodele opționale (earnings, insiders, macro) sunt folosite doar dacă există."""

    def prices(self, ticker: str, period: str) -> pd.DataFrame: ...
    def fundamentals(self, ticker: str) -> dict[str, Any]: ...
    def news(self, ticker: str) -> list[dict[str, Any]]: ...


class MarketData:
    def __init__(
        self,
        alpha_vantage_key: str | None = None,
        twelve_data_key: str | None = None,
        fred_key: str | None = None,
        sec_user_agent: str | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        self.cache = cache or DiskCache()
        av_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY") or None
        td_key = twelve_data_key or os.getenv("TWELVE_DATA_API_KEY") or None
        sec_ua = sec_user_agent or os.getenv("SEC_USER_AGENT") or None

        from .sources.alphavantage import AlphaVantage
        from .sources.fred import Fred
        from .sources.twelvedata import TwelveData

        self.av = AlphaVantage(av_key) if av_key else None
        self.td = TwelveData(td_key) if td_key else None
        self.fred = Fred(fred_key or os.getenv("FRED_API_KEY") or None, cache=self.cache)
        self.sec = None
        if sec_ua:
            from .sources.sec import SecEdgar

            try:
                self.sec = SecEdgar(sec_ua, cache=self.cache)
            except DataError as exc:
                log.warning("%s", exc)

    # ---- prețuri ----
    def prices(self, ticker: str, period: str) -> pd.DataFrame:
        key = f"prices_{ticker}"
        cached = self.cache.get_frame(key, max_age_hours=12)
        if cached is not None and not cached.empty:
            return cached
        frame = None
        if self.td:
            try:
                frame = self.td.daily(ticker)
            except DataError as exc:
                log.warning("%s; încerc Yahoo Finance", exc)
        if frame is None:
            frame = self._yahoo_prices(ticker, period)
        # Doar bare închise: în timpul ședinței, bara de azi are preț și volum parțiale.
        frame = drop_partial_bar(frame)
        self.cache.set_frame(key, frame)
        return frame

    def _yahoo_prices(self, ticker: str, period: str) -> pd.DataFrame:
        import yfinance as yf

        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        except Exception as exc:  # yfinance ridică tipuri diferite de la o versiune la alta
            raise DataError(f"Nu am putut descărca prețurile pentru {ticker}: {exc}") from exc
        if df.empty:
            raise DataError(f"Nu există prețuri pentru {ticker} (simbol greșit?)")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        return df

    def market(self, period: str) -> pd.DataFrame | None:
        try:
            return self.prices(MARKET_SYMBOL, period)
        except DataError as exc:
            log.warning("Trendul pieței indisponibil: %s", exc)
            return None

    # ---- companie ----
    def fundamentals(self, ticker: str) -> dict[str, Any]:
        if self.td:
            key = f"statistics_{ticker}"
            cached = self.cache.get_json(key, max_age_hours=24)
            if cached:
                return cached
            try:
                stats = self.td.statistics(ticker)
                if stats:
                    self.cache.set_json(key, stats)
                    return stats
            except DataError as exc:
                log.warning("%s; încerc Yahoo Finance", exc)
        import yfinance as yf

        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as exc:
            log.warning("Fundamentale indisponibile pentru %s: %s", ticker, exc)
            return {}
        return {k: info[k] for k in FUNDAMENTAL_KEYS if info.get(k) is not None}

    def news(self, ticker: str) -> list[dict[str, Any]]:
        if self.av:
            try:
                items = self.av.news(ticker)
                if items:
                    return items
            except DataError as exc:
                log.warning("%s", exc)
        return self._yahoo_headlines(ticker)

    def _earnings_report(self, ticker: str) -> dict[str, Any] | None:
        key = f"earnings2_{ticker}"
        cached = self.cache.get_json(key, max_age_hours=24 * 3)
        if isinstance(cached, dict):
            quarters = cached.get("quarters")
            return {"quarters": _restore_dates(quarters, "reported") if quarters is not None else None, "next": cached.get("next")}
        report = None
        if self.td is not None:
            try:
                report = self.td.earnings_report(ticker)
            except DataError as exc:
                log.warning("%s", exc)
        if report is None and self.av is not None:
            try:
                report = {"quarters": self.av.earnings(ticker), "next": None}
            except DataError as exc:
                log.warning("%s", exc)
        calendar = getattr(self.av, "next_earnings", None)
        if report is not None and report.get("next") is None and calendar is not None:
            # Twelve Data nu dă mereu raportul următor; calendarul Alpha Vantage costă o cerere la 3 zile.
            try:
                report = {**report, "next": calendar(ticker)}
            except DataError as exc:
                log.warning("%s", exc)
        if report is not None:
            nxt = report.get("next")
            report = {"quarters": report.get("quarters"), "next": nxt.isoformat() if hasattr(nxt, "isoformat") else nxt}
            self.cache.set_json(key, report)
        return report

    def earnings(self, ticker: str) -> list[dict[str, Any]] | None:
        report = self._earnings_report(ticker)
        return report["quarters"] if report else None

    def next_earnings(self, ticker: str) -> date | None:
        """Data următorului raport trimestrial (doar din Twelve Data)."""
        report = self._earnings_report(ticker)
        try:
            return date.fromisoformat(report["next"]) if report and report.get("next") else None
        except ValueError:
            return None

    def insiders(self, ticker: str, close: pd.Series | None = None) -> list[dict[str, Any]] | None:
        """Cumpărări (cod P) și vânzări (cod S) pe piață ale insiderilor."""
        key = f"insiders_{ticker}"
        cached = self.cache.get_json(key, max_age_hours=24)
        if cached is not None:
            return _restore_dates(cached, "date")
        trades = None
        if self.sec:
            try:
                trades = [t for t in self.sec.insider_trades(ticker) if t["code"] in ("P", "S")]
            except DataError as exc:
                log.warning("%s", exc)
        if trades is None and self.td:
            try:
                trades = self.td.insiders(ticker)
            except DataError as exc:
                log.warning("%s", exc)
        if trades is None and self.av and close is not None:
            from .sources.alphavantage import classify_insiders

            try:
                trades = classify_insiders(self.av.insiders(ticker), close)
            except DataError as exc:
                log.warning("%s", exc)
        if trades is not None:
            self.cache.set_json(key, trades)
        return trades

    def revisions(self, ticker: str) -> dict[str, Any] | None:
        """Reviziile estimărilor de profit ale analiștilor (Twelve Data), păstrate o zi."""
        if self.td is None:
            return None
        key = f"eps_trend_{ticker}"
        cached = self.cache.get_json(key, max_age_hours=24)
        if cached is not None:
            return cached or None
        try:
            trend = self.td.eps_trend(ticker)
        except DataError as exc:
            log.warning("%s", exc)
            return None
        self.cache.set_json(key, trend or {})
        return trend

    def macro(self) -> dict[str, Any] | None:
        snap = self.fred.snapshot()
        return snap or None

    def _yahoo_headlines(self, ticker: str) -> list[dict[str, Any]]:
        import yfinance as yf

        try:
            raw = yf.Ticker(ticker).news or []
        except Exception as exc:
            log.warning("Știri Yahoo indisponibile pentru %s: %s", ticker, exc)
            return []
        return parse_yahoo_news(raw)


def _restore_dates(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    from datetime import date

    out = []
    for r in rows:
        try:
            out.append({**r, field: date.fromisoformat(str(r[field])[:10])})
        except (KeyError, ValueError):
            continue
    return out


def parse_alpha_vantage_feed(feed: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    items = []
    for article in feed:
        match = next(
            (t for t in article.get("ticker_sentiment", []) if t.get("ticker", "").upper() == ticker.upper()),
            None,
        )
        items.append(
            {
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "published": article.get("time_published", ""),
                "summary": article.get("summary", ""),
                "sentiment": _to_float(match.get("ticker_sentiment_score")) if match else None,
                "relevance": _to_float(match.get("relevance_score")) if match else None,
            }
        )
    return items


def parse_yahoo_news(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for entry in raw:
        # yfinance >= 0.2.50 pune totul sub "content"; versiunile vechi nu.
        content = entry.get("content", entry)
        provider = content.get("provider")
        title = content.get("title")
        if not title:
            continue
        items.append(
            {
                "title": title,
                "source": provider.get("displayName", "") if isinstance(provider, dict) else content.get("publisher", ""),
                "published": content.get("pubDate") or content.get("providerPublishTime", ""),
                "summary": content.get("summary", ""),
                "sentiment": None,
                "relevance": None,
            }
        )
    return items


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
