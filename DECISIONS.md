# Decision log

Why things are the way they are, so future changes are informed. Newest first
within each phase.

## Phase 0 — Foundations

**D1 — Python 3.11, `uv`, src layout.** Matches the build environment; `uv` gives
fast, reproducible installs with a committed `uv.lock`. CI runs `uv sync
--frozen` so the lock is authoritative.

**D2 — Tooling: ruff (lint) + black (format) + pyright (types, `basic`).** Line
length 100; ruff owns import order and lint, black owns formatting (E501 ignored
in ruff to avoid fighting it). Pyright starts at `basic` to avoid drowning in
third-party (pandas/numpy) typing noise while still catching real errors; we can
tighten to `standard`/`strict` per-module as the code matures.

**D3 — One database, many jobs: Postgres + `pgvector`.** The bitemporal lake, the
feature store (later), and the episodic-memory vector store (Phase 3) all live in
one Postgres instance. At single-user scale this avoids operating a second
system. The `pgvector` extension is enabled by the initial migration even though
vectors arrive later.

**D4 — Purpose-built backtester, not zipline/backtrader/vectorbt.** The entire
point is to replay the world strictly through *knowledge time* from the
bitemporal store. Off-the-shelf engines make point-in-time integrity hard and can
hide lookahead. A small, well-tested, in-house engine makes honesty structural.

**D5 — Bitemporal model = (event_time, knowledge_time); store naive UTC.** The
two timestamps are the foundation of honesty. We deliberately do **not** require
`knowledge_time >= event_time`, because consensus estimates/forecasts are known
*before* the event they describe (the surprise-encoding layer depends on this).
Timestamps are persisted as **naive UTC** so behaviour is identical across
Postgres and SQLite; the application layer is always tz-aware UTC, normalised at
the storage boundary. Naive datetimes are rejected at the application edge — an
unlabelled timestamp is how lookahead sneaks in.

**D6 — In-memory store is the test oracle; SQL store is validated against it.**
Both implementations share the revision-collapsing and ordering logic, so they
return byte-identical results. The Hypothesis property test proves the
knowledge-time guarantee against both (and against real Postgres in CI). The
in-memory store is also what the harness uses in unit tests — fast and DB-free.

**D7 — Corrections are new revisions, surfaced only at their knowledge time.** A
restatement is a new `Observation` sharing a `logical_key` with a higher
`revision`; `as_of(K)` returns the latest revision *knowable by K*, never one
published later. Tested explicitly.

**D8 — Execution safety planted now, enforced later.** Settings default to
`execution_mode=paper` and `live_trading_enabled=false`; `assert_live_allowed()`
fails closed unless *both* switches are flipped. Live order routing (Phase 5)
will be coupled to hard caps, circuit-breakers, and a kill-switch. Per the user's
choice, execution capability is sequenced *earlier* than the brief's Phase 8, but
the safety primitives remain inseparable from it.

**D9 — SQLite as the local integration-test backend.** The SQLAlchemy store runs
on SQLite (in-memory, `StaticPool`) for fast local/CI tests without a daemon, and
on Postgres for production and CI's integration job. Alembic targets Postgres
(it enables `pgvector`); tests create the schema from ORM metadata.
