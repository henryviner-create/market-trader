"""Data-quality checks and the strict gateway."""

from __future__ import annotations

from datetime import date

import pytest

from market_trader.collectors import IngestionGateway, validate_observations
from market_trader.collectors.quality import DataQualityError
from market_trader.core.schema import Observation
from market_trader.core.time import day_close
from market_trader.storage import InMemoryBitemporalStore


def _price(entity_id: str, close: float) -> Observation:
    d = day_close(date(2023, 1, 3))
    return Observation(
        source="price",
        dataset="price.ohlcv",
        entity_type="equity",
        entity_id=entity_id,
        event_time=d,
        knowledge_time=d,
        value={"close": close},
    )


def test_clean_price_passes() -> None:
    assert validate_observations([_price("AAPL", 100.0)]).ok


def test_nan_and_nonpositive_prices_flagged() -> None:
    assert not validate_observations([_price("X", float("nan"))]).ok
    assert not validate_observations([_price("Y", 0.0)]).ok


def test_congress_negative_lag_flagged() -> None:
    bad = Observation(
        source="congress",
        dataset="disclosure.congress_trade",
        entity_type="equity",
        entity_id="X",
        event_time=day_close(date(2023, 2, 1)),
        knowledge_time=day_close(date(2023, 1, 1)),  # disclosure before transaction (impossible)
        value={},
    )
    report = validate_observations([bad])
    assert not report.ok
    assert any(i.check == "lag" for i in report.errors)


def test_strict_gateway_raises_and_lenient_gateway_warns() -> None:
    bad = [_price("X", float("nan"))]

    with pytest.raises(DataQualityError):
        IngestionGateway(InMemoryBitemporalStore(), strict=True).ingest(bad)

    lenient_store = InMemoryBitemporalStore()
    summary = IngestionGateway(lenient_store, strict=False).ingest(bad)
    assert summary.quality_ok is False
    assert lenient_store.count() == 1  # still stored, but flagged
