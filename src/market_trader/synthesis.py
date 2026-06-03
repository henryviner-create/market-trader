"""Synthesis: combine the quant score, episodic analogs, and events into a ranked
watchlist — each name carrying a confidence and the **explicit case against**.

The case-against is generated from the data (analog disagreement, downside tails,
missing corroboration, weak magnitude), enforcing the discipline that the system
must surface disconfirming evidence, not just a verdict. An LLM thesis can be
attached on top via the reasoning tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from market_trader.memory import detect_events
from market_trader.memory.episodic import EpisodicMemory
from market_trader.storage.bitemporal import BitemporalStore


@dataclass
class Recommendation:
    symbol: str
    quant_score: float
    confidence: float
    events: list[str] = field(default_factory=list)
    analog: dict[str, Any] = field(default_factory=dict)
    case_against: str = ""
    thesis: str | None = None


def _confidence(score: float, analog: dict[str, Any]) -> float:
    base = min(1.0, abs(score) / 2.0)
    if analog.get("n", 0) > 0:
        share = float(analog.get("share_positive", 0.5))
        base *= share if score > 0 else (1.0 - share)
    return round(float(base), 3)


def _case_against(score: float, analog: dict[str, Any], events: list[str]) -> str:
    parts: list[str] = []
    if analog.get("n", 0) > 0:
        share = float(analog.get("share_positive", 0.5))
        if (score > 0 and share < 0.6) or (score < 0 and share > 0.4):
            parts.append(f"analogs disagree ({int(share * 100)}% resolved positive)")
        if float(analog.get("q10", 0.0)) < 0:
            parts.append(f"downside tail q10={round(float(analog['q10']), 3)}")
    if not events:
        parts.append("no corroborating events")
    if abs(score) < 0.5:
        parts.append("weak signal magnitude")
    return "; ".join(parts) or "no major disconfirming evidence found"


def synthesize(
    store: BitemporalStore,
    as_of: datetime,
    scores: pd.Series,
    *,
    episodic: EpisodicMemory | None = None,
    query_vectors: dict[str, np.ndarray] | None = None,
    top_n: int = 5,
) -> list[Recommendation]:
    ranked = pd.Series(scores, dtype=float).dropna().sort_values(ascending=False)

    events_by_entity: dict[str, list[str]] = {}
    for event in detect_events(store, as_of):
        events_by_entity.setdefault(event.entity_id, []).append(str(event.event_type))

    recommendations: list[Recommendation] = []
    for symbol in ranked.head(top_n).index:
        sym = str(symbol)
        score = float(ranked[symbol])
        analog: dict[str, Any] = {}
        if episodic is not None and query_vectors and sym in query_vectors:
            analog = episodic.outcome_distribution(query_vectors[sym], k=10)
        events = events_by_entity.get(sym, [])
        recommendations.append(
            Recommendation(
                symbol=sym,
                quant_score=round(score, 4),
                confidence=_confidence(score, analog),
                events=events,
                analog=analog,
                case_against=_case_against(score, analog, events),
            )
        )
    return recommendations
