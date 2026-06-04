"""EdgarClient: offline Form-4 XML parsing + budgeted per-symbol fetch (injected transport)."""

from __future__ import annotations

import json
from datetime import date

from market_trader.collectors.edgar import EdgarClient, Form4Collector

_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>COOK TIMOTHY D</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><officerTitle>CEO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-05-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.5</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-05-02</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>500</value></transactionShares></transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_parse_form4_xml_reads_each_transaction() -> None:
    recs = EdgarClient.parse_form4_xml(_FORM4_XML, filing_date=date(2024, 5, 3))
    assert len(recs) == 2
    buy = recs[0]
    assert buy.issuer_ticker == "AAPL" and buy.insider_name == "COOK TIMOTHY D"
    assert buy.transaction_code == "P" and buy.insider_title == "CEO"
    assert buy.transaction_date == date(2024, 5, 1) and buy.filing_date == date(2024, 5, 3)
    assert buy.shares == 1000.0 and buy.price_per_share == 150.5
    # normalize() flags the purchase and records the ~2-day filing lag
    obs = Form4Collector().normalize(recs)
    assert obs[0].value["is_purchase"] is True and obs[1].value["is_purchase"] is False
    assert obs[0].metadata["filing_lag_days"] == 2


def test_parse_form4_xml_bad_input_is_empty() -> None:
    assert EdgarClient.parse_form4_xml("not xml at all", filing_date=date(2024, 1, 1)) == []


def test_fetch_for_symbols_walks_submissions_and_parses() -> None:
    tickers = json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
    subs = json.dumps(
        {
            "filings": {
                "recent": {
                    "form": ["4", "8-K", "4"],
                    "accessionNumber": ["0000320193-24-000077", "x", "0000320193-19-000001"],
                    "filingDate": [
                        date.today().isoformat(),
                        date.today().isoformat(),
                        "2019-01-01",  # outside the lookback window -> skipped
                    ],
                    "primaryDocument": ["xslF345X05/wk-form4.xml", "8k.htm", "old.xml"],
                }
            }
        }
    )

    def transport(url: str) -> str:
        if "company_tickers" in url:
            return tickers
        if "submissions" in url:
            return subs
        if url.endswith("wk-form4.xml"):
            return _FORM4_XML
        return ""

    client = EdgarClient(user_agent="test", transport=transport)
    recs = client.fetch_for_symbols(["AAPL", "ZZZZ"], lookback_days=30)

    assert recs and all(r.issuer_ticker == "AAPL" for r in recs)  # ZZZZ has no CIK -> skipped
    assert {r.transaction_code for r in recs} == {"P", "S"}  # 8-K filtered; 2019 out of window


def test_fetch_for_symbols_follows_filings_files_shards() -> None:
    # Older filings live in `filings.files` shards; the fetcher must follow the ones
    # whose date range overlaps the lookback (and skip shards that are entirely older).
    tickers = json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
    subs = json.dumps(
        {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0000320193-24-000077"],
                    "filingDate": [date.today().isoformat()],
                    "primaryDocument": ["recent-form4.xml"],
                },
                "files": [
                    {"name": "shard-recent.json", "filingFrom": "2021-01-01", "filingTo": "2022-12-31"},
                    {"name": "shard-ancient.json", "filingFrom": "2010-01-01", "filingTo": "2012-12-31"},
                ],
            }
        }
    )
    shard_recent = json.dumps(
        {
            "form": ["4", "8-K"],
            "accessionNumber": ["0000320193-22-000001", "x"],
            "filingDate": ["2022-06-01", "2022-06-02"],
            "primaryDocument": ["old-form4.xml", "8k.htm"],
        }
    )

    def transport(url: str) -> str:
        if "company_tickers" in url:
            return tickers
        if url.endswith("CIK0000320193.json"):
            return subs
        if url.endswith("shard-recent.json"):
            return shard_recent
        if url.endswith("shard-ancient.json"):
            raise AssertionError("ancient shard is outside the lookback and must not be fetched")
        if url.endswith(".xml"):
            return _FORM4_XML
        return ""

    client = EdgarClient(user_agent="test", transport=transport)
    recs = client.fetch_for_symbols(["AAPL"], lookback_days=2000)  # ~5.5y -> includes 2022 shard

    assert len(recs) == 4  # 2 txns from recent + 2 from the in-window shard
    assert {r.filing_date for r in recs} == {date.today(), date(2022, 6, 1)}
