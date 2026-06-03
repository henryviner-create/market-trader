"""The daily briefing: ranking, deterministic render, and LLM narration via a mock."""

from __future__ import annotations

from datetime import date

from market_trader.collectors import (
    Form4Collector,
    FredSeriesCollector,
    IngestionGateway,
    PriceCollector,
)
from market_trader.core.synthetic import business_days
from market_trader.core.time import day_close
from market_trader.reasoning import (
    MockLLMProvider,
    build_briefing_context,
    generate_llm_brief,
    render_brief_markdown,
)
from market_trader.storage import InMemoryBitemporalStore


def _store() -> tuple[InMemoryBitemporalStore, list]:
    store = InMemoryBitemporalStore()
    gw = IngestionGateway(store)
    days = business_days(date(2023, 1, 2), 70)
    for sym, series in [
        ("AAPL", [100 * (1.01**i) for i in range(70)]),  # clear winner
        ("MSFT", [100 * (0.99**i) for i in range(70)]),  # clear loser
    ]:
        gw.ingest(
            PriceCollector().normalize(
                [
                    {"date": d.isoformat(), "symbol": sym, "close": c}
                    for d, c in zip(days, series, strict=True)
                ]
            )
        )
    gw.ingest(
        Form4Collector().normalize(
            [
                {
                    "issuer_ticker": "AAPL",
                    "insider_name": "X",
                    "transaction_code": "P",
                    "transaction_date": "2023-03-01",
                    "filing_date": "2023-03-03",
                }
            ]
        )
    )
    gw.ingest(
        FredSeriesCollector("DGS10").normalize(
            [{"date": "2023-01-01", "realtime_start": "2023-02-01", "value": "4.0"}]
        )
    )
    return store, days


def test_context_ranks_winner_top_and_renders() -> None:
    store, days = _store()
    ctx = build_briefing_context(store, day_close(days[-1]), top_n=5)
    assert ctx.top_signals
    assert ctx.top_signals[0]["symbol"] == "AAPL"  # momentum + insider buy

    md = render_brief_markdown(ctx)
    assert "Pre-market briefing" in md
    assert "Regime" in md and "AAPL" in md
    assert "Not financial advice" in md


def test_llm_brief_calls_provider_with_data_and_discipline() -> None:
    store, days = _store()
    ctx = build_briefing_context(store, day_close(days[-1]))
    provider = MockLLMProvider(canned="THESIS")

    out = generate_llm_brief(ctx, provider)
    assert out == "THESIS"
    assert provider.calls
    system, prompt = provider.calls[0]
    assert "case against" in system.lower()  # discipline rule present
    assert "AAPL" in prompt  # point-in-time data reached the model
