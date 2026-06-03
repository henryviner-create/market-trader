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
| Execution | Broker abstraction (paper-default), hard gates for live | Phase 2 (paper) / Phase 5 (live) |

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

## Operations (target)

Local-first via `docker compose` (Postgres + pgvector). Phase 7 promotes to an
always-on Linux VM with Prefect orchestration, a Redis/RabbitMQ queue,
Prometheus + Grafana, data-freshness and drift alerts, `restart: always`
recovery, secrets management, and cost monitoring with LLM cadence gates.
