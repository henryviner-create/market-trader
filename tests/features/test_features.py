"""Feature families: correctness, flow counting, regime, and no lookahead."""

from __future__ import annotations

from datetime import date

from market_trader.collectors import (
    CongressTradesCollector,
    Form4Collector,
    FredSeriesCollector,
    IngestionGateway,
    PriceCollector,
)
from market_trader.core.synthetic import business_days
from market_trader.core.time import day_close
from market_trader.features import (
    CongressLeadershipBuys,
    FeatureStore,
    InsiderNetBuys,
    Momentum,
    default_features,
    macro_regime,
)
from market_trader.storage import InMemoryBitemporalStore


def _ingest_prices(
    store: InMemoryBitemporalStore, symbol: str, closes: list[float], n0: int = 0
) -> list:
    days = business_days(date(2023, 1, 2), n0 + len(closes))[n0:]
    recs = [
        {"date": d.isoformat(), "symbol": symbol, "close": c}
        for d, c in zip(days, closes, strict=True)
    ]
    IngestionGateway(store).ingest(PriceCollector().normalize(recs))
    return days


def test_momentum_ranks_winners_above_losers() -> None:
    store = InMemoryBitemporalStore()
    rising = [100 * (1.01**i) for i in range(70)]
    falling = [100 * (0.99**i) for i in range(70)]
    flat = [100.0] * 70
    _ingest_prices(store, "UP", rising)
    _ingest_prices(store, "DOWN", falling)
    days = _ingest_prices(store, "FLAT", flat)

    mom = Momentum(lookback=60).compute(store, day_close(days[-1]), ["UP", "DOWN", "FLAT"])
    assert mom["UP"] > mom["FLAT"] > mom["DOWN"]


def test_feature_ignores_data_after_as_of() -> None:
    store = InMemoryBitemporalStore()
    days = _ingest_prices(store, "UP", [100 * (1.01**i) for i in range(50)])
    as_of = day_close(days[-1])
    before = Momentum(lookback=20).compute(store, as_of, ["UP"])["UP"]

    # Insert a crash *after* as_of; the point-in-time feature must not change.
    _ingest_prices(store, "UP", [10.0] * 20, n0=50)
    after = Momentum(lookback=20).compute(store, as_of, ["UP"])["UP"]
    assert before == after


def test_insider_net_buys_counts_disclosed_purchases() -> None:
    store = InMemoryBitemporalStore()
    IngestionGateway(store).ingest(
        Form4Collector().normalize(
            [
                {
                    "issuer_ticker": "AAPL",
                    "insider_name": "A",
                    "transaction_code": "P",
                    "transaction_date": "2023-01-10",
                    "filing_date": "2023-01-12",
                },
                {
                    "issuer_ticker": "AAPL",
                    "insider_name": "B",
                    "transaction_code": "P",
                    "transaction_date": "2023-01-11",
                    "filing_date": "2023-01-13",
                },
                {
                    "issuer_ticker": "AAPL",
                    "insider_name": "C",
                    "transaction_code": "S",
                    "transaction_date": "2023-01-12",
                    "filing_date": "2023-01-14",
                },
            ]
        )
    )
    f = InsiderNetBuys(window_days=90).compute(store, day_close(date(2023, 2, 1)), ["AAPL", "MSFT"])
    assert f["AAPL"] == 1.0  # 2 buys - 1 sell
    assert f["MSFT"] == 0.0  # no activity is neutral, not missing


def test_congress_feature_only_counts_high_signal_roles() -> None:
    store = InMemoryBitemporalStore()
    IngestionGateway(store).ingest(
        CongressTradesCollector().normalize(
            [
                {
                    "representative": "Leader",
                    "chamber": "house",
                    "ticker": "AAPL",
                    "transaction_type": "buy",
                    "transaction_date": "2023-01-01",
                    "disclosure_date": "2023-02-15",
                    "role": "leadership",
                },
                {
                    "representative": "Backbench",
                    "chamber": "house",
                    "ticker": "MSFT",
                    "transaction_type": "buy",
                    "transaction_date": "2023-01-01",
                    "disclosure_date": "2023-02-15",
                    "role": "member",
                },
            ]
        )
    )
    f = CongressLeadershipBuys(window_days=300).compute(
        store, day_close(date(2023, 3, 1)), ["AAPL", "MSFT"]
    )
    assert f["AAPL"] == 1.0
    assert f["MSFT"] == 0.0  # backbencher is noise


def test_macro_regime_reads_yield_curve() -> None:
    store = InMemoryBitemporalStore()
    gw = IngestionGateway(store)
    gw.ingest(
        FredSeriesCollector("DGS10").normalize(
            [{"date": "2023-01-01", "realtime_start": "2023-02-01", "value": "4.0"}]
        )
    )
    gw.ingest(
        FredSeriesCollector("DGS2").normalize(
            [{"date": "2023-01-01", "realtime_start": "2023-02-01", "value": "3.0"}]
        )
    )
    regime = macro_regime(store, day_close(date(2023, 2, 15)))
    assert regime["yield_curve_slope"] == 1.0
    assert regime["risk_on"] is True and regime["label"] == "risk_on"


def test_feature_store_matrix_shape() -> None:
    store = InMemoryBitemporalStore()
    _ingest_prices(store, "UP", [100 * (1.01**i) for i in range(70)])
    _ingest_prices(store, "DOWN", [100 * (0.99**i) for i in range(70)])
    matrix = FeatureStore(store, default_features()).compute_matrix(
        day_close(date(2023, 4, 1)), ["UP", "DOWN"]
    )
    assert list(matrix.index) == ["UP", "DOWN"]
    assert {"mom_60", "vol_20", "insider_net_buys_90d"} <= set(matrix.columns)
