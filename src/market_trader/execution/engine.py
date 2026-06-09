"""The execution engine — turns target weights into broker orders, PAPER-first.

Order of operations (each a gate that can stop the flow):
  1. live gating  — if mode=live, ``assert_live_allowed`` (fails closed otherwise);
  2. kill switch  — halt if engaged;
  3. drawdown circuit-breaker — auto-engage the kill switch and halt on breach;
  4. capital ceiling — cap deployable capital;
  5. per-name + gross risk limits — refuse over-limit targets;
  6. order sanity + order-rate cap, then submit via the broker, audit-logging each.

Default mode is paper; the engine never enables live on its own.
"""

from __future__ import annotations

from datetime import datetime

from market_trader.backtest.types import Weights
from market_trader.config import Settings, get_settings
from market_trader.core.identity import with_deterministic_id
from market_trader.core.schema import Observation
from market_trader.core.time import utcnow
from market_trader.execution.audit import log_execution_event
from market_trader.execution.broker import Broker, BrokerError, Order, OrderSide
from market_trader.execution.guardrails import (
    KillSwitch,
    KillSwitchEngaged,
    OrderRateLimiter,
    OrderSanityError,
    sanity_check_order,
)
from market_trader.observability import get_logger
from market_trader.portfolio.risk import DrawdownCircuitBreaker, RiskLimits, check_order
from market_trader.storage.bitemporal import BitemporalStore

_log = get_logger("execution")

# Persisted high-water mark for the drawdown governor. A fresh ExecutionEngine (and breaker)
# is built every cycle, so without this the peak reseeds to the cycle's equity each run and the
# 25% governor can only ever see an intra-cycle drop. Persisting it lets the breaker measure
# drawdown from the true multi-day/multi-restart peak.
DRAWDOWN_STATE_DATASET = "risk.drawdown_state"


def _load_drawdown_state(store: BitemporalStore) -> tuple[float | None, bool]:
    rows = store.as_of(utcnow(), dataset=DRAWDOWN_STATE_DATASET)
    if not rows:
        return None, False
    latest = max(rows, key=lambda o: o.event_time)  # append-only log -> newest wins
    peak = latest.value.get("peak")
    return (float(peak) if peak else None), bool(latest.value.get("tripped", False))


class ExecutionEngine:
    def __init__(
        self,
        broker: Broker,
        *,
        settings: Settings | None = None,
        limits: RiskLimits | None = None,
        audit_store: BitemporalStore | None = None,
        kill_switch: KillSwitch | None = None,
        breaker: DrawdownCircuitBreaker | None = None,
        rate_limiter: OrderRateLimiter | None = None,
    ) -> None:
        self.broker = broker
        self.settings = settings or get_settings()
        self.limits = limits or RiskLimits()
        self.audit_store = audit_store
        self.kill_switch = kill_switch or KillSwitch()
        self.breaker = breaker or DrawdownCircuitBreaker(self.settings.max_drawdown_halt)
        # Restore the drawdown high-water mark so the governor survives the per-cycle engine
        # rebuild (only when we own the breaker and have somewhere to read it from).
        if breaker is None and self.audit_store is not None:
            peak, was_tripped = _load_drawdown_state(self.audit_store)
            if peak is not None:
                self.breaker.seed(peak, was_tripped)
        self.rate_limiter = rate_limiter or OrderRateLimiter(self.settings.max_orders_per_interval)

    def rebalance(
        self, target_weights: Weights, prices: dict[str, float], *, as_of: datetime
    ) -> list[Order]:
        if self.settings.execution_mode == "live":
            self.settings.assert_live_allowed()  # fails closed unless explicitly armed
        if self.kill_switch.engaged:
            raise KillSwitchEngaged(self.kill_switch.reason or "engaged")

        account = self.broker.get_account()
        tripped = self.breaker.update(account.equity)
        self._save_drawdown_state(as_of)  # persist the high-water mark across cycles
        if tripped:
            self.kill_switch.engage("drawdown_circuit_breaker")
            self._audit(
                as_of, "PORTFOLIO", "halt", {"reason": "drawdown", "peak": self.breaker.peak}
            )
            raise KillSwitchEngaged("drawdown circuit-breaker tripped")

        # Daily-loss kill: stop and require a human re-arm after a hard down day.
        # Off when max_daily_loss == 0. Compared against the previous close.
        if self.settings.max_daily_loss > 0 and account.last_equity > 0:
            day_return = account.equity / account.last_equity - 1.0
            if day_return <= -self.settings.max_daily_loss:
                self.kill_switch.engage("daily_loss_limit")
                self._audit(
                    as_of,
                    "PORTFOLIO",
                    "halt",
                    {"reason": "daily_loss", "day_return": round(day_return, 4)},
                )
                raise KillSwitchEngaged(f"daily loss {day_return:.2%} hit limit")

        deployable = min(account.equity, self.settings.capital_ceiling)
        for symbol, weight in target_weights.items():
            others = {k: v for k, v in target_weights.items() if k != symbol}
            check_order(symbol, weight, others, self.limits)  # refuses over-limit targets

        positions = {p.symbol: p.qty for p in self.broker.get_positions()}
        # Count still-open orders as effective holdings: without this, a re-run
        # before the prior order fills would see a stale (zero) position and
        # double-submit. Crucial for the intraday loop, harmless for daily.
        for pending in self.broker.get_open_orders():
            remaining = pending.qty - pending.filled_qty
            signed = remaining if pending.side == OrderSide.BUY else -remaining
            positions[pending.symbol] = positions.get(pending.symbol, 0.0) + signed
        # Build the orders first, then submit SELLs before BUYs: a sell frees
        # buying power, so a fully-invested rebalance can fund its entries instead
        # of being rejected (Alpaca returns 403 "insufficient buying power" for a
        # buy it cannot cover). Stable sort preserves intra-side order.
        planned: list[tuple[Order, float]] = []
        for symbol, weight in target_weights.items():
            price = prices.get(symbol)
            if not price or price <= 0:
                continue
            target_qty = weight * deployable / price
            current_qty = positions.get(symbol, 0.0)
            delta = target_qty - current_qty
            # No-trade band: don't churn an existing position for a small adjustment (turnover
            # is a real cost on a daily book, and the vol-governor rescales every name each
            # cycle). A full entry (no current position) or exit (target 0) always executes;
            # only mid-rebalance drifts inside the band are skipped.
            band = self.settings.rebalance_band
            if (
                band > 0
                and current_qty != 0
                and target_qty != 0
                and abs(delta) < band * abs(target_qty)
            ):
                continue
            qty = abs(delta)
            if not self.settings.fractional_shares:
                # Whole shares: many small-caps aren't fractionable on the broker, and a
                # fractional order on one 403s and aborts the rebalance. Floor toward zero so
                # a buy never overshoots buying power; the sub-share remainder is immaterial.
                qty = float(int(qty))
            if qty * price < 1.0:  # skip dust (and sub-one-share deltas in whole-share mode)
                continue
            order = Order(
                client_order_id=f"{as_of.strftime('%Y%m%dT%H%M%S')}-{symbol}",
                symbol=symbol,
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                qty=qty,
            )
            planned.append((order, price))
        planned.sort(key=lambda op: op[0].side != OrderSide.SELL)  # SELLs first

        orders: list[Order] = []
        skipped = 0
        for order, price in planned:
            try:
                sanity_check_order(order, ref_price=price)
            except OrderSanityError as exc:  # a malformed order never aborts the book
                skipped += 1
                _log.warning("order_skipped", symbol=order.symbol, reason=str(exc))
                self._audit(as_of, order.symbol, "order_skipped", {"reason": str(exc)})
                continue
            if not self.rate_limiter.allow():
                # Protective cap reached: stop submitting but keep the fills so far, rather
                # than aborting the whole rebalance (a large book must not lose its sells).
                _log.warning(
                    "order_rate_capped", cap=self.rate_limiter.max_orders, placed=len(orders)
                )
                self._audit(
                    as_of, "PORTFOLIO", "order_rate_capped", {"cap": self.rate_limiter.max_orders}
                )
                break
            try:
                filled = self.broker.submit_order(order)
            except BrokerError as exc:  # e.g. 403 insufficient buying power / not fractionable
                skipped += 1
                _log.warning(
                    "order_rejected", symbol=order.symbol, side=order.side.value, reason=str(exc)
                )
                self._audit(
                    as_of,
                    order.symbol,
                    "order_rejected",
                    {"side": order.side.value, "qty": order.qty, "reason": str(exc)},
                )
                continue  # the single bad name is skipped; the rest of the book still trades
            orders.append(filled)
            self._audit(
                as_of,
                order.symbol,
                "order",
                {
                    "client_order_id": order.client_order_id,
                    "side": order.side.value,
                    "qty": order.qty,
                    "status": filled.status.value,
                    "mode": self.settings.execution_mode,
                },
            )
        if skipped:
            self._audit(
                as_of, "PORTFOLIO", "rebalance_summary", {"placed": len(orders), "skipped": skipped}
            )
        return orders

    def halt_and_flatten(self, prices: dict[str, float], *, as_of: datetime) -> list[Order]:
        """Kill-switch action: engage and close all open positions."""
        self.kill_switch.engage("manual_kill")
        orders: list[Order] = []
        for pos in self.broker.get_positions():
            order = Order(
                client_order_id=f"flat-{as_of.strftime('%Y%m%dT%H%M%S')}-{pos.symbol}",
                symbol=pos.symbol,
                side=OrderSide.SELL if pos.qty > 0 else OrderSide.BUY,
                qty=abs(pos.qty),
            )
            orders.append(self.broker.submit_order(order))
            self._audit(as_of, pos.symbol, "flatten", {"client_order_id": order.client_order_id})
        return orders

    def _save_drawdown_state(self, as_of: datetime) -> None:
        """Persist the breaker's high-water mark so the next cycle's engine restores it."""
        peak = self.breaker.peak
        if self.audit_store is None or peak is None:
            return
        self.audit_store.upsert_many(
            [
                with_deterministic_id(
                    Observation(
                        source="execution",
                        dataset=DRAWDOWN_STATE_DATASET,
                        entity_type="portfolio",
                        entity_id="PORTFOLIO",
                        ref="drawdown",
                        event_time=as_of,
                        knowledge_time=as_of,
                        value={"peak": float(peak), "tripped": bool(self.breaker.tripped)},
                    )
                )
            ]
        )

    def _audit(self, as_of: datetime, symbol: str, event: str, detail: dict) -> None:
        if self.audit_store is not None:
            log_execution_event(
                self.audit_store, as_of=as_of, symbol=symbol, event=event, detail=detail
            )
