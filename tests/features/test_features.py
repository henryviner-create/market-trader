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
from market_trader.collectors.fundamentals import FundamentalsCollector
from market_trader.core.synthetic import business_days
from market_trader.core.time import day_close
from market_trader.features import (
    CongressLeadershipBuys,
    FeatureStore,
    InsiderNetBuys,
    Momentum,
    Volatility,
    candidate_features,
    default_features,
    macro_regime,
)
from market_trader.features.fundamental import EarningsSurprise, EarningsYield
from market_trader.storage import InMemoryBitemporalStore


def _ingest_eps(store: InMemoryBitemporalStore, ticker: str, quarters: list) -> None:
    """quarters: list of (period_end_iso, filed_iso, eps)."""
    recs = [
        {"ticker": ticker, "period_end": pe, "filed_date": fd, "eps": e} for pe, fd, e in quarters
    ]
    IngestionGateway(store).ingest(FundamentalsCollector().normalize(recs))


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


def test_opportunistic_filter_drops_scheduled_insiders() -> None:
    # "Sched" files every January for 4 years -> routine -> dropped; "Irreg" buys once
    # off-schedule -> opportunistic -> kept (Cohen-Malloy-Pomorski).
    recs = [
        {
            "issuer_ticker": t,
            "insider_name": n,
            "transaction_code": "P",
            "transaction_date": td,
            "filing_date": fd,
        }
        for t, n, td, fd in [
            ("RTN", "Sched", "2021-01-15", "2021-01-16"),
            ("RTN", "Sched", "2022-01-15", "2022-01-16"),
            ("RTN", "Sched", "2023-01-15", "2023-01-16"),
            ("RTN", "Sched", "2024-01-15", "2024-01-16"),  # in the 90d window
            ("OPP", "Irreg", "2023-12-10", "2023-12-11"),  # in the 90d window
        ]
    ]
    store = InMemoryBitemporalStore()
    IngestionGateway(store).ingest(Form4Collector().normalize(recs))
    as_of = day_close(date(2024, 2, 1))

    allf = InsiderNetBuys(window_days=90).compute(store, as_of, ["RTN", "OPP"])
    assert allf["RTN"] == 1.0 and allf["OPP"] == 1.0  # raw: both recent buys counted

    opp = InsiderNetBuys(window_days=90, opportunistic_only=True).compute(
        store, as_of, ["RTN", "OPP"]
    )
    assert opp["RTN"] == 0.0  # the scheduled January filer is stripped out
    assert opp["OPP"] == 1.0  # the irregular buyer is kept


def test_momentum_skip_excludes_the_recent_window() -> None:
    store = InMemoryBitemporalStore()
    closes = [100 * (1.01**i) for i in range(60)] + [50.0] * 10  # long uptrend, recent crash
    days = _ingest_prices(store, "X", closes)
    as_of = day_close(days[-1])

    plain = Momentum(lookback=50).compute(store, as_of, ["X"])["X"]
    skipped = Momentum(lookback=50, skip=10).compute(store, as_of, ["X"])["X"]
    assert plain < 0.0  # the recent crash drags raw momentum negative
    assert skipped > 0.0  # skipping the last 10 days restores the underlying uptrend


def test_low_volatility_factor_is_negative_volatility() -> None:
    store = InMemoryBitemporalStore()
    calm = [100 * (1.001**i) for i in range(40)]
    wild = [100.0]
    for i in range(39):
        wild.append(wild[-1] * (1.08 if i % 2 == 0 else 0.93))
    _ingest_prices(store, "CALM", calm)
    days = _ingest_prices(store, "WILD", wild)
    as_of = day_close(days[-1])

    vol = Volatility(window=30).compute(store, as_of, ["CALM", "WILD"])
    lowvol = Volatility(window=30, low_vol=True).compute(store, as_of, ["CALM", "WILD"])
    assert vol["WILD"] > vol["CALM"]  # WILD is the more volatile name
    assert lowvol["CALM"] > lowvol["WILD"]  # the low-vol factor flips the ranking
    assert abs(lowvol["CALM"] - (-vol["CALM"])) < 1e-12  # it is exactly -vol


def test_candidate_features_extends_default_with_gated_signals() -> None:
    names = {f.name for f in candidate_features()}
    assert {f.name for f in default_features()} <= names  # candidates are a superset
    assert {
        "insider_net_buys_90d_opp",
        "mom_252_skip21",
        "lowvol_120",
        "earnings_yield",
        "earnings_surprise",
    } <= names


def test_earnings_yield_is_ttm_eps_over_price() -> None:
    store = InMemoryBitemporalStore()
    _ingest_prices(store, "AAA", [100.0] * 5)  # latest price = 100
    _ingest_eps(
        store,
        "AAA",
        [
            ("2023-03-31", "2023-04-15", 1.0),
            ("2023-06-30", "2023-07-15", 1.0),
            ("2023-09-30", "2023-10-15", 1.0),
            ("2023-12-31", "2024-01-15", 1.0),  # TTM EPS = 4.0
        ],
    )
    y = EarningsYield().compute(store, day_close(date(2024, 6, 1)), ["AAA", "BBB"])
    assert abs(y["AAA"] - 0.04) < 1e-9  # 4.0 / 100
    assert y.isna()["BBB"]  # no fundamentals -> NaN (no opinion), not a misleading 0


def test_earnings_surprise_standardizes_the_latest_quarter() -> None:
    store = InMemoryBitemporalStore()
    quarters = [
        ("2022-03-31", "2022-04-15", 1.0),
        ("2022-06-30", "2022-07-15", 1.0),
        ("2022-09-30", "2022-10-15", 1.0),
        ("2022-12-31", "2023-01-15", 1.0),
        ("2023-03-31", "2023-04-15", 1.0),
        ("2023-06-30", "2023-07-15", 1.0),
        ("2023-09-30", "2023-10-15", 1.1),  # small prior surprises -> low std
        ("2023-12-31", "2024-01-15", 2.0),  # big jump vs a year earlier -> high SUE
    ]
    _ingest_eps(store, "AAA", quarters)

    sue = EarningsSurprise().compute(store, day_close(date(2024, 2, 1)), ["AAA"])
    assert sue["AAA"] > 3.0  # a large standardized surprise inside the drift window

    # well after the last filing -> outside the drift window -> no active signal
    stale = EarningsSurprise().compute(store, day_close(date(2024, 12, 1)), ["AAA"])
    assert stale.isna()["AAA"]
