# Architecture

A layered, event-driven system. Each tier is independently testable. This is a
living document; it tracks both the **target state** and **what exists today**.

## The one principle everything serves

Lookahead bias is the number-one cause of fake success, so the foundation is a
**bitemporal** data model: every fact carries two timestamps —

- **event time** — when it was true in the world;
- **knowledge time** — when it first became knowable to *us*.

Backtests and live inference may only ever see data at its knowledge time. A
congressional trade is stamped with its execution date *and* its ~45-day-later
disclosure date; the backtest sees it only at disclosure time. This single rule
prevents most lookahead bias, and it is enforced in storage and re-checked at the
harness level.

## Tiers (target state → current status)

| Tier | Purpose | Status |
|------|---------|--------|
| 1. Collection | API collectors, scrapers, streams → normalisation gateway | Phase 1 |
| 2. Storage | **Bitemporal lake**, feature store, raw archive, vector store | **Phase 0: bitemporal lake ✅** |
| 3. Signals | Decorrelated families: price, fundamental, flow, news, macro, alt | Phase 2 |
| 4. Forecasting | GBTs + regularised linear + simple TS; regime-aware stacking; calibration | Phase 4 |
| 5. Market memory | Event taxonomy, surprise encoding, event-study impact, episodic/analog, dual memory | Phase 3 |
| 6. Portfolio & risk | Vol-targeting, fractional Kelly, HRP/MV-shrinkage, caps, circuit-breakers | Phase 5 |
| 7. Backtesting & validation | Walk-forward + purged k-fold + embargo; knowledge-time replay; costs; baselines | **Phase 0: harness skeleton ✅** |
| 8. Feedback loop | Prediction logging, scoring, drift monitoring, retraining, signal pruning | Phase 6 |
| 9. Reasoning (LLM) | Sourced thesis per name, constrained by the event-study statistics | Phase 2+ |
| 10. Presentation | Dashboard, daily briefing, alerts | Phase 1+ |
| Execution | Broker-agnostic adapter (Alpaca), **paper-first**, strictly downstream of risk, all guardrails | **Phase 8 (paper)** / Phase 9 (gated live) |

## What Phase 0 delivers

```
Observation (canonical, bitemporal)
        │  normalised at the edge
        ▼
BitemporalStore.as_of(knowledge_time)        ← never reveals the future
   ├── InMemoryBitemporalStore   (reference / oracle)
   └── SqlAlchemyBitemporalStore (SQLite for tests, Postgres+pgvector for prod)
        │
        ▼
StorePriceView (point-in-time)  ──►  Strategy.target_weights()
        │                                   │
        ▼                                   ▼
Backtest engine: forward-return accounting, costs on turnover
        │
        ▼
Metrics (Sharpe/Sortino/Calmar/MDD/turnover/hit-rate/calibration, bootstrap CIs)
   vs. equal-weight & buy-and-hold baselines, net of costs
```

- **`core/`** — `Observation` (the canonical schema), the knowledge-time clock,
  and deterministic synthetic data.
- **`storage/`** — the `BitemporalStore` protocol, two implementations that share
  revision/ordering logic (so they cannot diverge), and Alembic migrations that
  enable `pgvector` for the later episodic-memory layer.
- **`backtest/`** — point-in-time views, leakage-resistant splitters
  (`walk_forward`, `PurgedKFold` with embargo), a transaction-cost model,
  metrics with calibration and bootstrap CIs, baseline strategies, and the
  engine that ties them together via knowledge-time replay.

## Validation strategy

Walk-forward **+ purged k-fold with embargo**, replayed strictly through
knowledge time; realistic costs/spread/slippage; every result benchmarked vs
equal-weight and buy-and-hold **net of costs** with confidence intervals;
validated across regimes, not just dates; calibration treated as first-class.
Backtest returns are treated as an upper bound. Dedicated leakage and bitemporal
test suites must pass for any pipeline to be considered valid.

## Execution tier (paper-first, human-gated)

Execution is the **final** stage and sits strictly downstream of risk:

```
Forecast → Score → Portfolio & Risk (sizing · limits · drawdown breakers)
        → Execution adapter (broker-agnostic; Alpaca default)
        → Broker API → fills/rejections → feedback loop
```

- Orders that breach any risk constraint are **refused in code before
  submission** — execution cannot bypass risk.
- A single `TRADING_MODE` control (`execution_mode`) defaults to **paper** and is
  deliberately hard to flip (two env switches + a startup confirmation). Paper and
  live share one code path; only base URL + keys differ.
- **Mandatory guardrails**, all built and tested in the paper phase: kill-switch,
  hard pre-trade limits (per-name size, gross/net notional, daily loss, order
  rate), drawdown circuit-breaker, order sanity checks, heartbeat/dead-man's
  switch, a low hard **capital ceiling**, and a full timestamped audit log.
- **Graduation gates (paper → live):** beat equal-weight and buy-and-hold net of
  costs across a meaningful multi-regime paper run, acceptable calibration, max
  drawdown within tolerance, no unresolved incident, all guardrails proven. Live
  is then a **separate, explicit, human-approved** step (Phase 9) that begins in
  a live-dry-run/log-only sub-mode. **The system never enables live on its own.**

## Deployment & operations

Ops is first-class from Phase 0, not bolted on. The engine is a stateful,
long-running service — **never serverless, never laptop-hosted**.

**Locked infrastructure (D11, D12):**

- **Host:** a persistent **DigitalOcean Basic Droplet** (4 vCPU / 8 GB / NVMe,
  Premium AMD/Intel preferred), **Ubuntu 24.04 LTS**, region nearest the operator
  (latency is immaterial). Scale-up trigger: a ≥16 GB Droplet if intraday / larger
  universe / heavy scraping / crypto are confirmed. Avoid DO App Platform and GPU
  Droplets.
- **Runtime:** Docker containers via `docker compose` (Kubernetes is overkill).
  Services: `db` (Postgres+pgvector), `migrate` (run-once), `engine` (long-running,
  health-checked). Healthchecks + `depends_on: service_healthy` gate startup;
  `restart: unless-stopped` + a systemd unit give crash/reboot self-healing.
- **Production LLM:** the **hosted Anthropic API** (no local LLM/GPU box). Claude
  Code is dev-time only; the engine calls the API itself on schedule, gated by
  cadence/cost limits. The key is a managed secret.
- **Twelve-factor:** config from env, **secrets never in repo/images**, structured
  JSON logs to stdout.

**Phase 0 baseline (now):** multi-stage non-root `Dockerfile`, `docker-compose`
(+ prod override), the `market-trader` CLI entrypoint, `.dockerignore`, a
beginner-friendly `scripts/bootstrap_vps.sh`, the systemd unit, and
`OPERATIONS.md`.

**Phase 7 hardening:** Prefect orchestration, a Redis/RabbitMQ queue, Prometheus
+ Grafana + Alertmanager and an external heartbeat, data-freshness and drift
alerts, **automated off-box backups with tested restores**, a Caddy reverse proxy
(auto-HTTPS) for any public dashboard, and gated CI/CD image build/push/deploy.

See [`OPERATIONS.md`](OPERATIONS.md) for the runbook and the staged VPS
provisioning walkthrough (delivered when deployment is reached or on request).
