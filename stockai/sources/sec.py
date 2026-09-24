"""SEC EDGAR (gratuit, oficial): tranzacțiile insiderilor din formularele 4.

SEC cere un antet User-Agent cu nume și email (setează SEC_USER_AGENT) și cel mult 10 cereri pe secundă.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Any

import requests

from ..cache import DiskCache, RateLimiter
from ..errors import DataError

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{name}"


class SecEdgar:
    def __init__(
        self,
        user_agent: str,
        session: requests.Session | None = None,
        cache: DiskCache | None = None,
        min_interval: float = 0.15,
    ) -> None:
        if "@" not in user_agent:
            raise DataError("SEC cere un User-Agent cu adresă de email (ex. 'StockAI nume@exemplu.ro')")
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.cache = cache or DiskCache()
        self.limiter = RateLimiter(min_interval)

    def _get(self, url: str) -> requests.Response:
        self.limiter.wait()
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise DataError(f"SEC EDGAR indisponibil: {exc}") from exc
        return resp

    def cik(self, ticker: str) -> int:
        table = self.cache.get_json("sec_company_tickers", max_age_hours=24 * 7)
        if table is None:
            try:
                table = self._get(TICKERS_URL).json()
            except ValueError as exc:
                raise DataError("SEC: listă de simboluri invalidă") from exc
            self.cache.set_json("sec_company_tickers", table)
        return lookup_cik(table, ticker)

    def insider_trades(self, ticker: str, days: int = 180, max_filings: int = 40) -> list[dict[str, Any]]:
        cik = self.cik(ticker)
        try:
            submissions = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        except ValueError as exc:
            raise DataError("SEC: răspuns invalid la lista de rapoarte") from exc
        trades: list[dict[str, Any]] = []
        for accession, name in recent_form4(submissions, date.today() - timedelta(days=days))[:max_filings]:
            url = ARCHIVE_URL.format(cik=cik, accession=accession.replace("-", ""), name=name)
            try:
                trades.extend(parse_form4(self._get(url).text))
            except (DataError, ET.ParseError):
                continue
        return trades


def lookup_cik(table: dict[str, Any], ticker: str) -> int:
    wanted = ticker.upper().replace(".", "-")
    for row in table.values():
        if str(row.get("ticker", "")).upper() == wanted:
            return int(row["cik_str"])
    raise DataError(f"SEC nu cunoaște simbolul {ticker}")


def recent_form4(submissions: dict[str, Any], since: date) -> list[tuple[str, str]]:
    """(număr de înregistrare, fișier XML) pentru formularele 4 depuse după `since`, cele mai noi primele."""
    recent = submissions.get("filings", {}).get("recent", {})
    out = []
    for form, filed, accession, doc in zip(
        recent.get("form", []), recent.get("filingDate", []), recent.get("accessionNumber", []), recent.get("primaryDocument", [])
    ):
        if form != "4" or not doc:
            continue
        try:
            if date.fromisoformat(filed) < since:
                continue
        except ValueError:
            continue
        # primaryDocument indică varianta formatată (xslF345X05/x.xml); XML-ul brut are același nume, în rădăcină.
        out.append((accession, doc.split("/")[-1]))
    return out


def _text(node: ET.Element | None, path: str) -> str:
    found = node.find(path) if node is not None else None
    return (found.text or "").strip() if found is not None and found.text else ""


def parse_form4(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    owner = root.find("reportingOwner")
    name = _text(owner, "reportingOwnerId/rptOwnerName")
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None
    title = _text(rel, "officerTitle") or ("Director" if _text(rel, "isDirector") in ("1", "true") else "")
    out = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        try:
            when = date.fromisoformat(_text(tx, "transactionDate/value")[:10])
            shares = float(_text(tx, "transactionAmounts/transactionShares/value") or 0)
            price = float(_text(tx, "transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        out.append(
            {
                "date": when,
                "owner": name,
                "title": title,
                "code": _text(tx, "transactionCoding/transactionCode"),
                "shares": shares,
                "price": price,
            }
        )
    return out
