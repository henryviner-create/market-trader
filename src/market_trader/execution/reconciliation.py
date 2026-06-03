"""Position reconciliation: intended vs. actual broker positions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from market_trader.execution.broker import Position


@dataclass(frozen=True)
class Divergence:
    symbol: str
    intended: float
    actual: float
    diff: float


def reconcile(
    intended: dict[str, float], broker_positions: Sequence[Position], *, tol: float = 1e-6
) -> list[Divergence]:
    """Return positions where intended quantity differs from the broker's."""
    actual = {p.symbol: p.qty for p in broker_positions}
    divergences: list[Divergence] = []
    for symbol in sorted(set(intended) | set(actual)):
        want = intended.get(symbol, 0.0)
        have = actual.get(symbol, 0.0)
        if abs(want - have) > tol:
            divergences.append(Divergence(symbol, want, have, want - have))
    return divergences
