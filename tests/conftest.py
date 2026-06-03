"""Shared test fixtures.

Every store-level test runs against the in-memory reference store and a SQLite
SQLAlchemy store. When ``TEST_DATABASE_URL`` (or ``MT_TEST_DATABASE_URL``) is set
— as it is in CI — the same tests also run against real Postgres, marked
``integration``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from market_trader.storage import InMemoryBitemporalStore
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore


def postgres_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL") or os.environ.get("MT_TEST_DATABASE_URL")


def make_store(kind: str) -> BitemporalStore:
    """Construct a fresh, empty store of the given kind."""
    if kind == "inmemory":
        return InMemoryBitemporalStore()
    if kind == "sqlite":
        # A private in-memory SQLite shared across sessions via a single pooled
        # connection — fast and isolated per call.
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
        store = SqlAlchemyBitemporalStore(engine)
        store.create_schema()
        return store
    if kind == "postgres":
        url = postgres_url()
        assert url, "postgres store requested without TEST_DATABASE_URL"
        store = SqlAlchemyBitemporalStore.from_url(url)
        store.drop_schema()
        store.create_schema()
        return store
    raise ValueError(f"unknown store kind: {kind}")


def _store_params() -> list:
    params = [pytest.param("inmemory", id="inmemory"), pytest.param("sqlite", id="sqlite")]
    if postgres_url():
        params.append(pytest.param("postgres", id="postgres", marks=pytest.mark.integration))
    return params


@pytest.fixture(params=_store_params())
def store(request: pytest.FixtureRequest) -> Iterator[BitemporalStore]:
    kind: str = request.param
    s = make_store(kind)
    yield s
    if kind == "postgres":
        s.drop_schema()  # type: ignore[attr-defined]


@pytest.fixture
def store_factory() -> Callable[[str], BitemporalStore]:
    """A stateless factory, safe to use inside Hypothesis property tests."""
    return make_store
