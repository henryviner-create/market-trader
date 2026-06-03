# market-trader

A **market intelligence & forecasting system** for a single sophisticated user —
built as **decision-support** that sharpens judgement by aggregating dispersed
public information and reasoning over it with explicit uncertainty.

> **This is not a guaranteed-return system and not a substitute for professional
> financial advice.** Backtests are upper bounds, not promises. The system is
> engineered to be honestly validated, not to look impressive.

## Status

| Phase | Scope | State |
|------:|-------|------|
| **0** | Foundations: bitemporal store, knowledge-time clock, validation/backtest harness, CI | **complete** |
| **1** | Collectors (EDGAR / FRED / prices / GDELT / Congress) + idempotent gateway + data-quality + PIT survivorship universe + dashboard | **complete** |
| **2** | Signal tier (feature store: technical/flow/regime) + LLM daily briefing | **complete** |
| **3** | Market-memory: taxonomy, surprise, event-study impact engine, episodic analogs | **complete** |
| **4** | Forecasting ensemble (logistic + GBT, regime-aware stacking, calibration) + purged-CV/OOS backtest | **complete** |
| **5** | Weighting (IC/decay/orthogonality) + portfolio/risk (vol-target, Kelly, HRP, limits, breaker) | **complete** |
| **6** | Synthesis (case-against) + feedback loop (prediction log, drift, pruning, alerts) | **complete** |
| **7** | 24/7 hardening: metrics, freshness, heartbeat, scheduler, monitoring profile, off-box backups | **complete** |
| **8** | Execution tier — **Alpaca paper only** + all guardrails | **complete** |
| 9 | Gated live consideration — human-approved, low capital ceiling | **gated (not enabled)** |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full target design and
[`DECISIONS.md`](DECISIONS.md) for the rationale behind key choices.

## The non-negotiables this foundation encodes

- **Optimise risk-adjusted return, not hit-rate.** The harness reports
  Sharpe / Sortino / Calmar / max-drawdown / turnover / calibration, always
  **net of costs** and **against equal-weight and buy-and-hold** baselines.
- **Point-in-time integrity.** Every fact carries an *event time* and a
  *knowledge time*; nothing is ever observable before its knowledge time. The
  rule is enforced in storage and independently re-checked at the harness level
  (see `tests/bitemporal/` and `tests/leakage/`).
- **Beat the dumb baseline or simplify.**

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Docker is optional
(only for Postgres).

```bash
uv sync          # create venv + install everything
make demo        # end-to-end Phase 0 demo on synthetic data
make check       # lint + types + fast tests (what CI runs, minus integration)
make test        # fast tests, no database required
```

With Postgres (the production SQL store + integration tests):

```bash
make db-up       # start pgvector Postgres via docker compose
make migrate     # apply Alembic migrations
export MT_TEST_DATABASE_URL=postgresql+psycopg://market:market@localhost:5432/market_trader
make test-int    # all tests, including the real-Postgres path
```

## Layout

```
src/market_trader/
  core/          canonical schema, knowledge-time clock, synthetic data
  storage/       bitemporal store (in-memory + SQLAlchemy) + Alembic migrations
  backtest/      validation harness: PIT views, splitters, costs, metrics, engine
  config/        pydantic settings (secrets from the environment)
  observability/ structured logging
scripts/         runnable demos
tests/           unit + bitemporal property + leakage suites
```

## Run with Docker / deploy

The engine is containerised and runs the whole stack with one command:

```bash
docker compose up -d --build     # db + migrations + engine (health-checked)
curl -s localhost:8080/health    # {"status":"ok","db":true,...}
```

**Production** runs **paper-first** on a persistent **DigitalOcean Droplet
(8 GB / 4 vCPU) / Ubuntu 24.04**, auto-recovering across crashes and reboots
(systemd + `restart` policies).
The reasoning layer uses the **hosted Anthropic API** (no local LLM). See
[`OPERATIONS.md`](OPERATIONS.md) for the runbook; a staged, beginner-friendly
provisioning walkthrough is delivered when you reach deployment or just ask.

## Execution posture (paper-first, human-gated)

The system is **paper-first**: every default, config, and example defaults to
paper, and no default can place a live order. Execution is the **last tier**,
strictly downstream of the risk layer.

Live trading is **disabled** until *both* `MT_EXECUTION_MODE=live` and
`MT_LIVE_TRADING_ENABLED=true` are set (`Settings.assert_live_allowed()` fails
closed otherwise) — and even then only after the paper→live **graduation gates**
are met and a human explicitly approves. **The system will never enable live on
its own.** The execution tier and all mandatory guardrails (kill-switch, hard
pre-trade limits, drawdown circuit-breaker, heartbeat/dead-man's switch, capital
ceiling, audit log) are built and proven in **Phase 8 (paper only)**; gated live
consideration is **Phase 9**. See `DECISIONS.md` D10.
