"""Market-memory layer — the experiential engine.

Cross-cutting "what usually happens after X" memory:

* **taxonomy** — a structured event vocabulary + rule-based detection.
* **surprise** — actual-vs-consensus encoding (the delta markets price on).
* **event_study** — market-model abnormal returns (AR/CAR) and the *distribution*
  of outcomes per event type, with significance.
* **episodic** — analog retrieval: the k nearest historical situations and how
  they resolved (explainable, good for rare events).

Together the episodic store and the parametric models (Phase 4) form a dual
memory that covers each other's weaknesses.
"""

from market_trader.memory.episodic import Episode, EpisodicMemory
from market_trader.memory.event_study import (
    EventOutcomeDistribution,
    MarketModel,
    aggregate_event_study,
    estimate_market_model,
    event_car,
)
from market_trader.memory.surprise import Surprise, encode_surprise
from market_trader.memory.taxonomy import Event, EventType, detect_events

__all__ = [
    "Episode",
    "EpisodicMemory",
    "Event",
    "EventOutcomeDistribution",
    "EventType",
    "MarketModel",
    "Surprise",
    "aggregate_event_study",
    "detect_events",
    "encode_surprise",
    "estimate_market_model",
    "event_car",
]
