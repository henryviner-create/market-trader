"""Point-in-time universe must include delisted names at the right times."""

from __future__ import annotations

from datetime import date

from market_trader.universe import PointInTimeUniverse


def test_seed_loads_and_contains_delisted_names() -> None:
    universe = PointInTimeUniverse.from_seed()
    assert "AAPL" in universe.all_symbols()
    assert len(universe.delisted()) >= 1
    assert "SIVB" in universe.delisted()


def test_delisted_name_is_point_in_time() -> None:
    universe = PointInTimeUniverse.from_seed()

    # SVB (SIVB) was removed 2023-03-17: a member before, gone after.
    assert universe.is_member_on("SIVB", date(2023, 1, 15)) is True
    assert universe.is_member_on("SIVB", date(2023, 4, 1)) is False
    assert "SIVB" in universe.members_on(date(2023, 1, 15))
    assert "SIVB" not in universe.members_on(date(2023, 4, 1))


def test_active_name_persists_and_membership_predates_addition() -> None:
    universe = PointInTimeUniverse.from_seed()
    assert "AAPL" in universe.members_on(date(2023, 4, 1))
    # Lehman was never a member after its 2008 collapse.
    assert universe.is_member_on("LEH", date(2010, 1, 1)) is False
    assert universe.is_member_on("LEH", date(2007, 1, 1)) is True
