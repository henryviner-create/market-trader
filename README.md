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
| **0** | Foundations: bitemporal store, canonical schema, knowledge-time clock, validation/backtest harness, CI | **complete** |
| 1 | MVP collectors (EDGAR / FRED / prices / news / congress) + dashboard | next |
| 2–7 | Signals, market-memory, forecasting, weighting, risk, feedback, 24/7 | planned |
| 8 | Execution tier — **Alpaca paper only** + all guardrails (graduation gates) | planned |
| 9 | Gated live consideration — human-approved, low capital ceiling | gated |

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
