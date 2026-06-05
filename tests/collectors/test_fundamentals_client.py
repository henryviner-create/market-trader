"""FundamentalsClient: offline companyfacts parsing + budgeted fetch (injected transport)."""

from __future__ import annotations

import json

from market_trader.collectors.fundamentals import FundamentalsClient

_COMPANYFACTS = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [
                        # Q1 2023 (~quarter) — the original announcement
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 1.50,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2023-04-28",
                        },
                        # same Q1 re-reported a year later as a comparative -> earlier filing wins
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 1.50,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2024-04-30",
                        },
                        # Q2 2023
                        {
                            "start": "2023-04-01",
                            "end": "2023-06-30",
                            "val": 1.20,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2023-07-28",
                        },
                        # year-to-date (6 months) -> not a quarter -> dropped
                        {
                            "start": "2023-01-01",
                            "end": "2023-06-30",
                            "val": 2.70,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2023-07-28",
                        },
                        # full year (10-K) -> dropped
                        {
                            "start": "2023-01-01",
                            "end": "2023-12-31",
                            "val": 6.00,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2024-02-01",
                        },
                    ]
                }
            }
        }
    },
}


def test_parse_companyfacts_keeps_quarterly_eps_earliest_filing() -> None:
    recs = FundamentalsClient.parse_companyfacts(_COMPANYFACTS, fallback_ticker="AAPL")
    by_end = {r.period_end.isoformat(): r for r in recs}

    assert set(by_end) == {"2023-03-31", "2023-06-30"}  # only the true quarters
    assert by_end["2023-03-31"].eps == 1.50
    # the original announcement date, not the later comparative filing
    assert by_end["2023-03-31"].filed_date.isoformat() == "2023-04-28"
    assert by_end["2023-06-30"].eps == 1.20
    assert all(r.ticker == "AAPL" for r in recs)


def test_fetch_for_symbols_walks_companyfacts() -> None:
    tickers = json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
    facts = json.dumps(_COMPANYFACTS)

    def transport(url: str) -> str:
        if "company_tickers" in url:
            return tickers
        if "companyfacts/CIK0000320193" in url:
            return facts
        return "{}"

    client = FundamentalsClient(user_agent="test", transport=transport)
    recs = client.fetch_for_symbols(["AAPL", "ZZZZ"])  # ZZZZ has no CIK -> skipped

    assert recs and all(r.ticker == "AAPL" for r in recs)
    assert {r.period_end.isoformat() for r in recs} == {"2023-03-31", "2023-06-30"}
