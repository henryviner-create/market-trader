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
fails closed unless *both* switches are flipped. Live order routing will be
coupled to hard caps, circuit-breakers, and a kill-switch. See **D10** for the
execution-tier policy (paper-first, human-gated).

**D9 — SQLite as the local integration-test backend.** The SQLAlchemy store runs
on SQLite (in-memory, `StaticPool`) for fast local/CI tests without a daemon, and
on Postgres for production and CI's integration job. Alembic targets Postgres
(it enables `pgvector`); tests create the schema from ORM metadata.

**D10 — Execution is paper-first and human-gated (supersedes the earlier
"wire live early").** Adopting the execution add-on module. The execution tier is
the *last* stage of the pipeline, strictly **downstream of the risk layer**
(orders that breach a risk constraint are refused in code before submission), and
is built and validated against **Alpaca paper** first — same code path as live,
only the base URL and key set differ. Live trading is a separate, later,
explicitly **human-approved** decision gated behind measurable paper graduation
criteria: beats equal-weight and buy-and-hold net of costs across a meaningful
multi-regime paper period, acceptable calibration, max drawdown within tolerance,
no unresolved operational incident, and all guardrails implemented + tested. The
paper→live switch defaults to paper and needs two env switches **plus** a startup
confirmation; **the system must never enable live autonomously** — if asked, it
prints the gate checklist and a risk warning and requires explicit confirmation.
Mandatory guardrails (kill-switch, hard pre-trade limits, drawdown
circuit-breaker, order sanity checks, heartbeat/dead-man's switch, low capital
ceiling, full audit log) are built and proven in the **paper** phase before live
is even an option. Roadmap change: **Phase 8 = execution tier, paper only**
(acceptance = the graduation gates); **Phase 9 (new) = gated live consideration**,
starting in a live-dry-run/log-only sub-mode, then a tiny capital ceiling. This
reverses the interim "wire live early" decision; **no broker scaffold is
introduced in Phase 2** anymore.

**D11 — Deployment is first-class from Phase 0; 24/7 via auto-recovery.** Adopting
the deployment add-on module. The engine is a stateful, long-running service —
**never serverless, never laptop-hosted**. It runs as Docker containers
orchestrated by `docker compose` (Kubernetes is overkill at single-user scale),
with per-service healthchecks, `depends_on: service_healthy` gating, named
volumes, and `restart: unless-stopped`. A systemd unit brings the stack up on
boot so crashes *and* reboots self-heal unattended. Twelve-factor: config from
env, **secrets never in the repo or images**, structured JSON logs to stdout.
The Phase 0 baseline (multi-stage non-root Dockerfile, compose with the
`migrate`/`engine` services, the `market-trader` CLI entrypoint, `.dockerignore`,
bootstrap script, `OPERATIONS.md`) is in place now; the full monitoring stack
(Prometheus/Grafana/Alertmanager + external heartbeat) and **automated off-box
backups with tested restores** are hardened in Phase 7. A dashboard may later
deploy separately (e.g. Vercel) reading the DB/API; the engine's control surfaces
are never publicly reachable.

**D12 — Infrastructure locked: Hetzner CX32 / Ubuntu 24.04 + hosted Anthropic
API.** Adopting the infrastructure add-on module. Host = a persistent **Hetzner
Cloud CX32** (4 vCPU / 8 GB / ~80 GB NVMe), **Ubuntu 24.04 LTS** — the cost-
optimal EU baseline; latency is immaterial for swing-horizon US equities, so we
do not pay to sit near the broker. Scale-up trigger to **CX42** (8 vCPU / 16 GB)
only if intraday, a larger universe, heavy scraping, or crypto are confirmed.
Production reasoning/labelling/synthesis call the **hosted Anthropic API** — **no
local LLM** on the VPS (avoids an expensive GPU box). Claude Code is a dev-time
tool, not the production runtime: the deployed engine calls the API itself, on
schedule, gated by cadence/cost limits with token-usage logging. The Anthropic
key is a managed, rotatable secret (`MT_ANTHROPIC_API_KEY`), never committed. A
staged, beginner-friendly provisioning walkthrough is delivered when the
deployment phase is reached or on request.
