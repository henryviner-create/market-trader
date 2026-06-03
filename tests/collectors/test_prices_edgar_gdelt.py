"""Prices / EDGAR Form 4 / GDELT normalisation and knowledge-time stamping."""

from __future__ import annotations

from datetime import date

from market_trader.collectors import Form4Collector, GdeltNewsCollector, PriceCollector
from market_trader.core.synthetic import PRICE_DATASET


def test_price_bar_lands_in_the_backtester_dataset() -> None:
    (o,) = PriceCollector().normalize(
        [{"date": "2023-01-03", "symbol": "aapl", "close": 125.0, "volume": 1_000_000}]
    )
    assert o.dataset == PRICE_DATASET  # same dataset the harness reads
    assert o.entity_id == "AAPL"
    assert o.event_time.date() == date(2023, 1, 3)
    assert o.knowledge_time == o.event_time  # known at the close
    assert o.value["close"] == 125.0
    assert o.value["volume"] == 1_000_000.0


def test_form4_two_day_lag_and_purchase_flag() -> None:
    (o,) = Form4Collector().normalize(
        [
            {
                "issuer_ticker": "nvda",
                "insider_name": "Jensen Huang",
                "insider_title": "CEO",
                "transaction_code": "P",
                "transaction_date": "2023-01-10",
                "filing_date": "2023-01-12",
                "shares": 10000,
                "price_per_share": 150.0,
            }
        ]
    )
    assert o.dataset == "filing.form4"
    assert o.event_time.date() == date(2023, 1, 10)
    assert o.knowledge_time.date() == date(2023, 1, 12)
    assert o.metadata["filing_lag_days"] == 2  # far more tradable than Congress
    assert o.value["is_purchase"] is True


def test_gdelt_links_entity_or_falls_back_to_global() -> None:
    linked, unlinked = GdeltNewsCollector().normalize(
        [
            {"seendate": "2023-02-01", "title": "Apple beats", "symbol": "AAPL", "tone": 4.2},
            {"seendate": "2023-02-01", "title": "Markets wobble", "tone": -1.0},
        ]
    )
    assert linked.entity_type == "equity" and linked.entity_id == "AAPL"
    assert linked.knowledge_time == linked.event_time
    assert unlinked.entity_type == "news_global" and unlinked.entity_id == "GLOBAL"
