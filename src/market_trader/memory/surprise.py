"""Surprise encoding.

Markets react to the *delta from what was priced in*, not the raw print. We store
actual-vs-consensus surprise (sign + magnitude), standardised by the historical
surprise dispersion where available.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Surprise:
    actual: float
    consensus: float
    surprise: float
    standardized: float | None
    direction: int  # +1 beat, -1 miss, 0 in line


def encode_surprise(
    actual: float, consensus: float, *, dispersion: float | None = None
) -> Surprise:
    delta = actual - consensus
    standardized = delta / dispersion if dispersion and dispersion > 0 else None
    direction = 1 if delta > 0 else (-1 if delta < 0 else 0)
    return Surprise(
        actual=actual,
        consensus=consensus,
        surprise=delta,
        standardized=standardized,
        direction=direction,
    )
