"""Point-in-time universe membership.

Survivorship bias is a top failure mode: a universe of *today's* index members,
applied to the past, quietly deletes every company that blew up or got acquired.
So membership is bitemporal too — each constituent has an ``added`` and an
optional ``removed`` date, and :meth:`members_on` answers "who was in the index on
date D", including names since delisted.

The bundled seed is a small *sample* (not the full S&P 500) that deliberately
includes delisted names so survivorship handling is testable. A
survivorship-correct full constituent history is wired in from a data source
later.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from typing import Any

_SEED_RESOURCE = ("market_trader.universe", "seed_constituents.json")


@dataclass(frozen=True)
class Constituent:
    symbol: str
    name: str
    added: date
    removed: date | None

    def active_on(self, on: date) -> bool:
        return self.added <= on and (self.removed is None or on < self.removed)


class PointInTimeUniverse:
    def __init__(self, constituents: Iterable[Constituent]) -> None:
        self._constituents = list(constituents)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> PointInTimeUniverse:
        cons = [
            Constituent(
                symbol=str(r["symbol"]).upper(),
                name=str(r["name"]),
                added=date.fromisoformat(str(r["added"])),
                removed=(date.fromisoformat(str(r["removed"])) if r.get("removed") else None),
            )
            for r in records
        ]
        return cls(cons)

    @classmethod
    def from_json_str(cls, text: str) -> PointInTimeUniverse:
        return cls.from_records(json.loads(text)["constituents"])

    @classmethod
    def from_seed(cls) -> PointInTimeUniverse:
        text = files(_SEED_RESOURCE[0]).joinpath(_SEED_RESOURCE[1]).read_text(encoding="utf-8")
        return cls.from_json_str(text)

    def members_on(self, on: date) -> list[str]:
        return sorted(c.symbol for c in self._constituents if c.active_on(on))

    def is_member_on(self, symbol: str, on: date) -> bool:
        s = symbol.upper()
        return any(c.symbol == s and c.active_on(on) for c in self._constituents)

    def all_symbols(self) -> list[str]:
        return sorted({c.symbol for c in self._constituents})

    def delisted(self) -> list[str]:
        return sorted(c.symbol for c in self._constituents if c.removed is not None)
