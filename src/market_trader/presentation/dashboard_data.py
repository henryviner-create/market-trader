"""Dashboard data layer (pure, tested).

Builds everything the dashboard renders *as of* a knowledge time, straight from
the bitemporal store — so the dashboard can only ever show what was knowable
then. The Streamlit shell is a thin renderer over this.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from market_trader.collectors.congress import CONGRESS_DATASET
from market_trader.collectors.edgar import FORM4_DATASET
from market_trader.collectors.fred import FRED_DATASET
from market_trader.collectors.gdelt import NEWS_DATASET
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.universe import PointInTimeUniverse


@dataclass
class DashboardData:
    as_of: datetime
    watchlist: list[str]
    latest_prices: dict[str, float]
    recent_insider: list[dict[str, Any]]
    recent_congress: list[dict[str, Any]]
    recent_news: list[dict[str, Any]]
    macro: dict[str, float]


def _summarise(obs: Any, *, keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        "entity_id": obs.entity_id,
        "event_date": obs.event_time.date().isoformat(),
        "knowledge_date": obs.knowledge_time.date().isoformat(),
        **{k: obs.value.get(k) for k in keys},
    }


def build_dashboard_data(
    store: BitemporalStore,
    as_of: datetime,
    *,
    universe: PointInTimeUniverse | None = None,
    limit: int = 25,
) -> DashboardData:
    universe = universe or PointInTimeUniverse.from_seed()

    latest_prices: dict[str, float] = {}
    for o in store.as_of(as_of, dataset=PRICE_DATASET):  # sorted by knowledge_time → last wins
        close = o.value.get("close")
        if close is not None:
            latest_prices[o.entity_id] = float(close)

    macro: dict[str, float] = {}
    for o in store.as_of(as_of, dataset=FRED_DATASET):
        v = o.value.get("value")
        if v is not None:
            macro[o.entity_id] = float(v)

    insider = [
        _summarise(o, keys=("transaction_code", "is_purchase", "insider_name"))
        for o in store.as_of(as_of, dataset=FORM4_DATASET)
    ][-limit:]
    congress = [
        _summarise(o, keys=("transaction_type", "representative", "amount_high"))
        for o in store.as_of(as_of, dataset=CONGRESS_DATASET)
    ][-limit:]
    news = [
        _summarise(o, keys=("title", "tone", "source"))
        for o in store.as_of(as_of, dataset=NEWS_DATASET)
    ][-limit:]

    return DashboardData(
        as_of=as_of,
        watchlist=universe.members_on(as_of.date()),
        latest_prices=latest_prices,
        recent_insider=insider,
        recent_congress=congress,
        recent_news=news,
        macro=macro,
    )
