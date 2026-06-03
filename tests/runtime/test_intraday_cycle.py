"""Intraday cycle: minute bars -> minute-momentum signals -> paper orders, offline.

A canned minute-bar payload is fed through a real AlpacaDataClient (injected
transport, no network) into an in-memory store + paper broker, proving one pass of
the live loop scores on minute bars and rebalances — reusing the daily
``run_paper_cycle`` path unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from market_trader.collectors.alpaca import AlpacaDataClient
from market_trader.config import Settings
from market_trader.execution.broker import OrderSide
from market_trader.execution.paper_broker import PaperBroker
from market_trader.runtime.intraday import run_intraday_cycle
from market_trader.storage import InMemoryBitemporalStore

_BASE = datetime(2026, 6, 3, 13, 30, tzinfo=UTC)


def _minute_payload(symbols: list[str], n: int = 45) -> tuple[dict[str, Any], dict[str, float]]:
    """Synthetic minute bars: earlier symbols trend up, later ones down."""
    bars: dict[str, list[dict[str, Any]]] = {}
    finals: dict[str, float] = {}
    mid = len(symbols) / 2.0
    for i, sym in enumerate(symbols):
        drift = 0.0015 * (mid - i)  # S0 strongest up, last strongest down
        px = 100.0
        rows: list[dict[str, Any]] = []
        for m in range(n):
            px *= 1.0 + drift
            ts = (_BASE + timedelta(minutes=m)).isoformat().replace("+00:00", "Z")
            rows.append({"t": ts, "o": px, "h": px, "l": px, "c": px, "v": 1000})
        bars[sym] = rows
        finals[sym] = px
    return {"bars": bars, "next_page_token": None}, finals


def _settings() -> Settings:
    return Settings(
        execution_mode="paper",
        capital_ceiling=10_000.0,
        intraday_momentum_lookback=30,
        intraday_meanrev_lookback=10,
        intraday_vol_window=20,
    )


def test_run_intraday_cycle_scores_minute_bars_and_trades() -> None:
    symbols = [f"S{i}" for i in range(8)]
    payload, finals = _minute_payload(symbols, n=45)

    def transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        return 200, payload

    data_client = AlpacaDataClient("k", "s", transport=transport)
    store = InMemoryBitemporalStore()
    broker = PaperBroker(finals, starting_cash=100_000.0)

    result = run_intraday_cycle(
        _settings(),
        watchlist=symbols,
        store=store,
        broker=broker,
        data_client=data_client,
        as_of=_BASE + timedelta(minutes=50),
        lookback_minutes=180,
    )

    assert result.scores  # minute-resolution features produced a ranking
    assert len(result.target_weights) == 2  # top_quantile of 8 -> 2 winners
    assert result.orders  # and it rebalanced on the broker
    assert all(o.side == OrderSide.BUY for o in result.orders)  # flat -> building into winners
    assert all(w <= _settings().max_position_weight + 1e-9 for w in result.target_weights.values())


def test_run_intraday_cycle_requires_keys_when_clients_not_injected() -> None:
    no_keys = Settings(execution_mode="paper", alpaca_key_id=None, alpaca_secret_key=None)
    with pytest.raises(RuntimeError):  # must build real clients -> keys required, fails fast
        run_intraday_cycle(no_keys)
