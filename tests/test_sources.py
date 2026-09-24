from datetime import date, timedelta

import pandas as pd
import pytest

from stockai.cache import DiskCache
from stockai.data import MarketData
from stockai.errors import DataError
from stockai.sources.alphavantage import classify_insiders, parse_earnings, parse_insiders
from stockai.sources.fred import parse_fred_csv, parse_fred_json
from stockai.sources.sec import lookup_cik, parse_form4, recent_form4
from stockai.sources.twelvedata import parse_time_series

# Forme reale, scurtate, ale răspunsurilor Alpha Vantage (observate prin conector).
AV_EARNINGS = {
    "symbol": "AAPL",
    "quarterlyEarnings": [
        {"fiscalDateEnding": "2026-03-31", "reportedDate": "2026-04-30", "reportedEPS": "2.01",
         "estimatedEPS": "1.94", "surprise": "0.07", "surprisePercentage": "3.6082", "reportTime": "post-market"},
        {"fiscalDateEnding": "2026-06-30", "reportedDate": "2026-07-30", "reportedEPS": "2.02",
         "estimatedEPS": "1.88", "surprise": "0.14", "surprisePercentage": "7.4468", "reportTime": "post-market"},
        {"fiscalDateEnding": "2003-09-30", "reportedDate": "2003-10-15", "reportedEPS": "0.01",
         "estimatedEPS": "None", "surprise": "0", "surprisePercentage": "None", "reportTime": "post-market"},
    ],
}
AV_INSIDERS = {
    "data": [
        {"transaction_date": "2026-09-15", "ticker": "AAPL", "executive": "NEWSTEAD, JENNIFER",
         "executive_title": "SVP", "security_type": "Common Stock", "acquisition_or_disposal": "D",
         "shares": "16228.0", "share_price": "331.34"},
        {"transaction_date": "2026-09-15", "ticker": "AAPL", "executive": "NEWSTEAD, JENNIFER",
         "executive_title": "SVP", "security_type": "Common Stock", "acquisition_or_disposal": "A",
         "shares": "30104.0", "share_price": ""},
        {"transaction_date": "2026-09-01", "ticker": "AAPL", "executive": "TERNUS, JOHN",
         "executive_title": "Director, CEO", "security_type": "Restricted Stock Unit",
         "acquisition_or_disposal": "A", "shares": "7690.0", "share_price": "0.0"},
        {"transaction_date": "2026-09-02", "ticker": "AAPL", "executive": "DOE, JANE",
         "executive_title": "Director", "security_type": "Common Stock", "acquisition_or_disposal": "A",
         "shares": "1000.0", "share_price": "320.00"},
        {"transaction_date": "2026-09-03", "ticker": "AAPL", "executive": "ROE, RICHARD",
         "executive_title": "CFO", "security_type": "Common Stock", "acquisition_or_disposal": "A",
         "shares": "5000.0", "share_price": "95.00"},
    ]
}

FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerCik>0000320193</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0001</rptOwnerCik><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-09-02</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>320.5</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-09-05</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>200</value></transactionShares>
        <transactionPricePerShare><value></value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_twelve_data_parses_values_oldest_first():
    payload = {
        "meta": {"symbol": "AAPL", "interval": "1day"},
        "values": [
            {"datetime": "2026-09-23", "open": "341.0", "high": "341.8", "low": "335.5", "close": "337.02", "volume": "31658823"},
            {"datetime": "2026-09-22", "open": "340.1", "high": "345.3", "low": "338.7", "close": "339.75", "volume": "40711786"},
        ],
        "status": "ok",
    }
    frame = parse_time_series(payload, "AAPL")
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame.index[0] < frame.index[1]
    assert frame["Close"].iloc[-1] == 337.02


def test_twelve_data_error_raises():
    with pytest.raises(DataError, match="run out of API credits"):
        parse_time_series({"code": 429, "message": "You have run out of API credits", "status": "error"}, "AAPL")


def test_earnings_sorted_newest_first_with_missing_surprise():
    quarters = parse_earnings(AV_EARNINGS)
    assert [q["reported"] for q in quarters] == [date(2026, 7, 30), date(2026, 4, 30), date(2003, 10, 15)]
    assert quarters[0]["surprise_pct"] == pytest.approx(7.4468)
    assert quarters[-1]["surprise_pct"] is None and quarters[-1]["estimate"] is None


def test_insiders_classification_keeps_only_market_trades():
    rows = parse_insiders(AV_INSIDERS)
    assert len(rows) == 5
    close = pd.Series([318.0, 331.0], index=pd.to_datetime(["2026-09-01", "2026-09-15"]))
    trades = classify_insiders(rows, close)
    # vânzarea la preț de piață și cumpărarea la 320 rămân; acordarea (preț gol), RSU-ul și exercitarea la 95 nu.
    assert [(t["owner"], t["code"]) for t in trades] == [("NEWSTEAD, JENNIFER", "S"), ("DOE, JANE", "P")]


def test_sec_lookup_and_form4_parsing():
    table = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
             "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway"}}
    assert lookup_cik(table, "aapl") == 320193
    assert lookup_cik(table, "BRK.B") == 1067983
    with pytest.raises(DataError):
        lookup_cik(table, "NOPE")

    trades = parse_form4(FORM4)
    assert trades[0] == {"date": date(2026, 9, 2), "owner": "Doe Jane", "title": "Director", "code": "P",
                         "shares": 1000.0, "price": 320.5}
    assert trades[1]["code"] == "M" and trades[1]["price"] == 0.0


def test_recent_form4_filters_by_form_and_date():
    today = date.today()
    subs = {"filings": {"recent": {
        "form": ["4", "10-Q", "4", "4"],
        "filingDate": [str(today), str(today), str(today - timedelta(days=400)), str(today - timedelta(days=3))],
        "accessionNumber": ["0001-26-000001", "0001-26-000002", "0001-25-000003", "0001-26-000004"],
        "primaryDocument": ["xslF345X05/wf-form4_1.xml", "q.htm", "xslF345X05/old.xml", "form4.xml"],
    }}}
    assert recent_form4(subs, today - timedelta(days=180)) == [("0001-26-000001", "wf-form4_1.xml"), ("0001-26-000004", "form4.xml")]


def test_fred_parsers():
    s = parse_fred_json({"observations": [{"date": "2026-09-22", "value": "0.55"}, {"date": "2026-09-23", "value": "."}]})
    assert s.to_dict() == {pd.Timestamp("2026-09-22"): 0.55}
    for header in ("DATE", "observation_date"):
        csv = f"{header},VIXCLS\n2026-09-21,17.1\n2026-09-22,.\n2026-09-23,18.4\n"
        s = parse_fred_csv(csv)
        assert list(s.values) == [17.1, 18.4]


class _FakeAV:
    def __init__(self):
        self.calls = 0

    def earnings(self, symbol):
        self.calls += 1
        return parse_earnings(AV_EARNINGS)


def test_market_data_caches_earnings_on_disk(tmp_path):
    md = MarketData(cache=DiskCache(tmp_path))
    md.av = _FakeAV()
    first = md.earnings("AAPL")
    second = md.earnings("AAPL")
    assert md.av.calls == 1
    assert second[0]["reported"] == first[0]["reported"] == date(2026, 7, 30)


def test_market_data_without_keys_returns_none_for_optional_sources(tmp_path):
    md = MarketData(cache=DiskCache(tmp_path))
    assert md.earnings("AAPL") is None
    assert md.insiders("AAPL") is None


def test_twelve_data_statistics_earnings_insiders():
    from stockai.scoring import fundamental_score
    from stockai.sources.twelvedata import parse_earnings as td_earnings
    from stockai.sources.twelvedata import parse_insiders as td_insiders
    from stockai.sources.twelvedata import parse_statistics

    # Forme observate prin conectorul Twelve Data (API-ul REST întoarce același JSON).
    stats = parse_statistics({
        "meta": {"symbol": "AAPL", "name": "Apple Inc."},
        "statistics": {
            "valuations_metrics": {"market_capitalization": 4.9e12, "trailing_pe": 38.9, "forward_pe": 35.4},
            "financials": {"profit_margin": 0.276, "income_statement": {"quarterly_revenue_growth": 0.164,
                                                                        "quarterly_earnings_growth_yoy": 0.271},
                           "balance_sheet": {"total_debt_to_equity_mrq": 78.4}},
            "stock_statistics": {"short_ratio": 2.97, "short_percent_of_shares_outstanding": 0.0096},
            "stock_price_summary": {"fifty_two_week_low": 243.4, "fifty_two_week_high": 345.3},
        },
    })
    assert stats["shortName"] == "Apple Inc." and stats["debtToEquity"] == 78.4 and stats["shortPercent"] == 0.0096
    assert fundamental_score(stats) > 0
    assert fundamental_score({**stats, "shortPercent": 0.15}) < fundamental_score(stats)

    quarters = td_earnings({"earnings": [
        {"date": "2099-10-29", "time": "After Hours", "eps_estimate": 2.1, "eps_actual": None, "surprise_prc": None},
        {"date": "2026-04-30", "time": "After Hours", "eps_estimate": 1.94, "eps_actual": 2.01, "surprise_prc": 3.61},
        {"date": "2026-07-30", "time": "After Hours", "eps_estimate": 1.89, "eps_actual": 2.02, "surprise_prc": 6.88},
    ]})
    assert [q["reported"] for q in quarters] == [date(2026, 7, 30), date(2026, 4, 30)]
    assert quarters[0]["surprise_pct"] == 6.88

    trades = td_insiders({"insider_transactions": [
        {"full_name": "NEWSTEAD JENNIFER", "position": "Officer", "date_reported": "2026-09-15", "shares": 1438,
         "value": 474813, "description": "Sale at price 330.19 per share."},
        {"full_name": "NEWSTEAD JENNIFER", "position": "Officer", "date_reported": "2026-09-15", "shares": 30104,
         "value": None, "description": ""},
        {"full_name": "DOE JANE", "position": "Director", "date_reported": "2026-09-10", "shares": 1000,
         "value": 150000, "description": "Purchase at price 150.00 per share."},
        {"full_name": "X", "position": "Director", "date_reported": "2026-09-01", "shares": 50,
         "value": 7000, "description": "Stock Gift at price 140.00 per share."},
        {"full_name": "Y", "position": "Officer", "date_reported": "2026-08-01", "shares": 200,
         "value": 60000, "description": "Sale at price 290.00 - 310.00 per share."},
    ]})
    assert [(t["owner"], t["code"], t["price"]) for t in trades] == [
        ("NEWSTEAD JENNIFER", "S", 330.19), ("DOE JANE", "P", 150.0), ("Y", "S", 290.0)]
