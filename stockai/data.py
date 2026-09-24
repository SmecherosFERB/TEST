"""Surse de date: prețuri și fundamentale (Yahoo Finance), știri cu sentiment (Alpha Vantage)."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import pandas as pd
import requests

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


class DataError(RuntimeError):
    pass


class DataProvider(Protocol):
    def prices(self, ticker: str, period: str) -> pd.DataFrame: ...
    def fundamentals(self, ticker: str) -> dict[str, Any]: ...
    def news(self, ticker: str) -> list[dict[str, Any]]: ...


class MarketData:
    """Yahoo Finance pentru prețuri și fundamentale; Alpha Vantage pentru sentiment, dacă are cheie."""

    def __init__(self, alpha_vantage_key: str | None = None) -> None:
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY") or None

    def prices(self, ticker: str, period: str) -> pd.DataFrame:
        import yfinance as yf

        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        except Exception as exc:  # yfinance ridică tipuri diferite de la o versiune la alta
            raise DataError(f"Nu am putut descărca prețurile pentru {ticker}: {exc}") from exc
        if df.empty:
            raise DataError(f"Nu există prețuri pentru {ticker} (simbol greșit?)")
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def fundamentals(self, ticker: str) -> dict[str, Any]:
        import yfinance as yf

        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as exc:
            log.warning("Fundamentale indisponibile pentru %s: %s", ticker, exc)
            return {}
        return {k: info[k] for k in FUNDAMENTAL_KEYS if info.get(k) is not None}

    def news(self, ticker: str) -> list[dict[str, Any]]:
        if self.alpha_vantage_key:
            items = self._alpha_vantage_news(ticker)
            if items:
                return items
        return self._yahoo_headlines(ticker)

    def _alpha_vantage_news(self, ticker: str, limit: int = 30) -> list[dict[str, Any]]:
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "limit": limit,
                    "apikey": self.alpha_vantage_key,
                },
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("Alpha Vantage indisponibil pentru %s: %s", ticker, exc)
            return []
        if "feed" not in payload:
            # La limită de apeluri, Alpha Vantage răspunde cu „Note” sau „Information”.
            log.warning("Alpha Vantage: %s", payload.get("Note") or payload.get("Information") or payload)
            return []
        return parse_alpha_vantage_feed(payload["feed"], ticker)

    def _yahoo_headlines(self, ticker: str) -> list[dict[str, Any]]:
        import yfinance as yf

        try:
            raw = yf.Ticker(ticker).news or []
        except Exception as exc:
            log.warning("Știri Yahoo indisponibile pentru %s: %s", ticker, exc)
            return []
        return parse_yahoo_news(raw)


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
