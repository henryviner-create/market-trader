"""Episodic memory & analog retrieval.

Each historical episode is a feature/context vector plus its realised outcome. On
a new situation we retrieve the k nearest analogs and surface *their outcome
distribution* ("the 12 closest situations resolved +3% median, with this
dispersion and these two drawdowns"). Explainable, and strong on rare events.

In-memory (numpy) here; production backs this with pgvector for scale. Vectors
should be pre-standardised by the caller so Euclidean distance is meaningful.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Episode:
    key: str
    vector: NDArray[np.float64]
    outcome: float
    regime: str | None = None
    event_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=float)


class EpisodicMemory:
    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    def __len__(self) -> int:
        return len(self._episodes)

    def add(self, episode: Episode) -> None:
        self._episodes.append(episode)

    def add_many(self, episodes: Iterable[Episode]) -> None:
        for e in episodes:
            self.add(e)

    def retrieve(
        self,
        query: NDArray[np.float64],
        k: int = 5,
        *,
        regime: str | None = None,
        event_type: str | None = None,
    ) -> list[tuple[Episode, float]]:
        q = np.asarray(query, dtype=float)
        scored = [
            (e, float(np.linalg.norm(e.vector - q)))
            for e in self._episodes
            if (regime is None or e.regime == regime)
            and (event_type is None or e.event_type == event_type)
        ]
        scored.sort(key=lambda pair: pair[1])
        return scored[:k]

    def outcome_distribution(
        self,
        query: NDArray[np.float64],
        k: int = 5,
        *,
        regime: str | None = None,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        neighbours = self.retrieve(query, k, regime=regime, event_type=event_type)
        outcomes: NDArray[np.float64] = np.array([e.outcome for e, _ in neighbours], dtype=float)
        if outcomes.size == 0:
            return {
                "n": 0,
                "mean": 0.0,
                "median": 0.0,
                "q10": 0.0,
                "q90": 0.0,
                "share_positive": 0.0,
            }
        return {
            "n": int(outcomes.size),
            "mean": float(outcomes.mean()),
            "median": float(np.median(outcomes)),
            "q10": float(np.quantile(outcomes, 0.1)),
            "q90": float(np.quantile(outcomes, 0.9)),
            "share_positive": float((outcomes > 0).mean()),
            "neighbours": [e.key for e, _ in neighbours],
        }
