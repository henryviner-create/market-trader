# storage — the bitemporal data lake

This tier decides whether the whole system is honest. Its one rule: **no consumer
may observe a fact before its `knowledge_time`.**

## The model

Everything is an `Observation` (see `core/schema.py`) carrying two clocks:

- `event_time` — when the fact was true in the world.
- `knowledge_time` — when it first became knowable to us.

Reads go through `as_of(knowledge_time, ...)`, which returns only facts with
`knowledge_time <= K`. Filters (`source`, `dataset`, `entity_type`, `entity_id`)
narrow the query; `latest_revision_only` (default) collapses corrections.

```python
from market_trader.storage import InMemoryBitemporalStore

store = InMemoryBitemporalStore()
store.add_many(observations)
visible_on_jan_20 = store.as_of(day_close(date(2022, 1, 20)))
```

## Corrections & restatements

A correction is a new `Observation` with the same `logical_key`
(`source, dataset, entity_id, event_time`) and a higher `revision`. `as_of(K)`
returns the most recent revision **knowable by K** — never one published later.
So a price restated on Feb 1 is invisible to a Jan 20 query and visible to a
Feb 5 one.

## Two implementations, one behaviour

- `InMemoryBitemporalStore` — the simple reference implementation and the test
  oracle.
- `SqlAlchemyBitemporalStore` — runs on **SQLite** (fast, daemon-free tests) and
  **Postgres + pgvector** (production). The point-in-time filter runs in SQL;
  revision collapsing and ordering reuse the shared helpers in `bitemporal.py`,
  so results are byte-identical to the oracle.

Timestamps persist as **naive UTC** (see `DECISIONS.md` D5) for cross-engine
consistency; the application layer is always tz-aware UTC.

## Migrations

Alembic (`alembic upgrade head`) targets Postgres and enables `pgvector` for the
Phase 3 episodic-memory layer. Tests build the schema from ORM metadata instead.

## Guarantees are tested, not assumed

- `tests/bitemporal/test_knowledge_time_guarantee.py` — a Hypothesis property
  test proving `as_of(K)` never reveals the future, across in-memory + SQLite
  (+ Postgres in CI).
- `tests/bitemporal/test_store_equivalence.py` — the SQL store matches the oracle
  and corrections respect knowledge time.
