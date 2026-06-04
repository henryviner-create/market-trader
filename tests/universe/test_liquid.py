"""The broad tradable universe and the selection resolver."""

from __future__ import annotations

from market_trader.universe.liquid import (
    EUROPE_ETFS,
    GLOBAL_LIQUID,
    LIQUID_LARGE_CAP,
    MEGACAP_WATCHLIST,
    resolve_universe,
)


def test_liquid_universe_is_broad_clean_and_diversified() -> None:
    assert len(LIQUID_LARGE_CAP) >= 100  # genuinely broad, not a handful
    assert len(set(LIQUID_LARGE_CAP)) == len(LIQUID_LARGE_CAP)  # no duplicates
    assert all(s.isupper() and "." not in s and s.isalpha() for s in LIQUID_LARGE_CAP)
    # reaches well beyond megacap tech into other sectors
    assert {"XOM", "JNJ", "JPM", "NEE", "CAT", "PLD", "PG", "DUK"} <= set(LIQUID_LARGE_CAP)


def test_resolve_universe_modes() -> None:
    assert resolve_universe("liquid") == LIQUID_LARGE_CAP
    assert resolve_universe("broad") == LIQUID_LARGE_CAP
    assert resolve_universe("") == LIQUID_LARGE_CAP  # default is broad
    assert resolve_universe("watchlist") == MEGACAP_WATCHLIST
    assert resolve_universe("megacap") == MEGACAP_WATCHLIST
    assert resolve_universe("aapl, msft ,nvda") == ["AAPL", "MSFT", "NVDA"]  # custom CSV


def test_global_universe_adds_tradable_european_exposure() -> None:
    assert set(LIQUID_LARGE_CAP) <= set(GLOBAL_LIQUID)  # superset of the US set
    assert {"ASML", "SAP", "NVO", "SHEL", "AZN", "UL"} <= set(GLOBAL_LIQUID)  # European ADRs
    assert set(EUROPE_ETFS) <= set(GLOBAL_LIQUID)  # Europe ETFs
    assert len(set(GLOBAL_LIQUID)) == len(GLOBAL_LIQUID)  # no duplicates with US names
    # all US-listed tickers (no ".PA"/".DE" — those aren't tradable on Alpaca)
    assert all(s.isupper() and "." not in s and s.isalpha() for s in GLOBAL_LIQUID)
    assert len(GLOBAL_LIQUID) >= 130  # enough breadth to fill a 50-name book


def test_resolve_universe_global_mode() -> None:
    assert resolve_universe("global") == GLOBAL_LIQUID
    assert resolve_universe("world") == GLOBAL_LIQUID


def test_resolve_universe_returns_a_copy() -> None:
    got = resolve_universe("liquid")
    got.append("ZZZZ")
    assert "ZZZZ" not in LIQUID_LARGE_CAP  # callers can't mutate the module constant
