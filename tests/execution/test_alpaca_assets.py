"""AlpacaBroker.list_us_equities: filter the assets feed to tradable symbols (offline)."""

from __future__ import annotations

import pytest

from market_trader.execution.alpaca import AlpacaBroker


def test_list_us_equities_keeps_only_tradable_named_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = AlpacaBroker("k", "s")
    assets = [
        {"symbol": "CCC", "tradable": True},
        {"symbol": "AAA", "tradable": True},
        {"symbol": "BBB", "tradable": False},  # not tradable -> dropped
        {"tradable": True},  # no symbol -> dropped
    ]
    monkeypatch.setattr(broker, "_request", lambda *a, **k: assets)

    assert broker.list_us_equities() == ["AAA", "CCC"]  # sorted, tradable, symbol present
