"""LLM-extracted signal collector — persists Opus's structured reads as point-in-time data.

The breadth factory's output lands here. An :class:`ExtractedSignal` becomes a normal
:class:`Observation` in the ``llm.signal`` dataset, with ``knowledge_time`` set to the run
time (>= every source headline's date, so there is no lookahead). A downstream
:class:`~market_trader.features.llm.LLMNewsSentiment` Feature reads it like any other, and
it must clear the same IC gate as every signal before it is trusted.

``extract_signals_for_universe`` is the orchestration: gather each name's recent headlines
from the news dataset, ask the (budgeted) provider for a structured score, and collect the
results — stopping cleanly when the LLM budget is spent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from market_trader.collectors.base import Collector
from market_trader.collectors.gdelt import NEWS_DATASET
from market_trader.core.schema import Observation
from market_trader.observability import get_logger
from market_trader.reasoning.budget import LLMBudgetExceeded
from market_trader.reasoning.extraction import ExtractedSignal, extract_news_signal
from market_trader.reasoning.llm import LLMProvider
from market_trader.storage.bitemporal import BitemporalStore

LLM_SIGNAL_DATASET = "llm.signal"

_log = get_logger("llm_features")


class LLMSignalCollector(Collector):
    source = "llm"
    parser_version = 1

    def normalize(self, raw: Any) -> list[Observation]:
        """``raw`` is an iterable of ``(ExtractedSignal, knowledge_time)`` pairs."""
        out: list[Observation] = []
        for sig, kt in raw:
            out.append(
                Observation(
                    source=self.source,
                    dataset=LLM_SIGNAL_DATASET,
                    entity_type="equity",
                    entity_id=sig.symbol.upper(),
                    ref=f"{sig.symbol.upper()}:{kt.date()}",
                    event_time=kt,
                    knowledge_time=kt,
                    value={
                        "score": sig.score,
                        "confidence": sig.confidence,
                        "rationale": sig.rationale,
                    },
                    metadata={"parser_version": self.parser_version},
                )
            )
        return out


def headlines_by_symbol(
    store: BitemporalStore, as_of: datetime, symbols: Sequence[str], *, window_days: int = 7
) -> dict[str, list[str]]:
    """Group recent (by knowledge_time) news titles per symbol — the LLM's reading material."""
    cutoff = as_of - timedelta(days=window_days)
    wanted = {s.upper() for s in symbols}
    out: dict[str, list[str]] = defaultdict(list)
    for o in store.as_of(as_of, dataset=NEWS_DATASET):
        if o.knowledge_time <= cutoff or o.entity_id not in wanted:
            continue
        title = o.value.get("title")
        if title:
            out[o.entity_id].append(str(title))
    return dict(out)


def extract_signals_for_universe(
    store: BitemporalStore,
    provider: LLMProvider,
    symbols: Sequence[str],
    as_of: datetime,
    *,
    window_days: int = 7,
) -> list[tuple[ExtractedSignal, datetime]]:
    """Extract a sentiment signal per name with recent news, bounded by the provider budget.

    Returns ``(signal, knowledge_time)`` pairs ready for :meth:`LLMSignalCollector.normalize`.
    A name with no headlines or an unparseable reply is simply skipped (neutral); when the
    LLM budget is exhausted the sweep stops and returns what it has so far (resumable).
    """
    heads = headlines_by_symbol(store, as_of, symbols, window_days=window_days)
    out: list[tuple[ExtractedSignal, datetime]] = []
    for sym in symbols:
        hs = heads.get(sym.upper())
        if not hs:
            continue
        try:
            sig = extract_news_signal(provider, symbol=sym.upper(), headlines=hs)
        except LLMBudgetExceeded:
            _log.warning("llm_budget_exhausted", extracted=len(out), of=len(heads))
            break
        if sig is not None:
            out.append((sig, as_of))
    _log.info("llm_extract_sweep", extracted=len(out), with_news=len(heads))
    return out
