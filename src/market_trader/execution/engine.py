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
from market_trader.execution.audit import log_execution_event
from market_trader.execution.broker import Broker, Order, OrderSide
from market_trader.execution.guardrails import (
    KillSwitch,
    KillSwitchEngaged,
    OrderRateExceeded,
    OrderRateLimiter,
    sanity_check_order,
)
from market_trader.observability import get_logger
from market_trader.portfolio.risk import DrawdownCircuitBreaker, RiskLimits, check_order
from market_trader.storage.bitemporal import BitemporalStore

_log = get_logger("execution")


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
        self.rate_limiter = rate_limiter or OrderRateLimiter(self.settings.max_orders_per_interval)

    def rebalance(
        self, target_weights: Weights, prices: dict[str, float], *, as_of: datetime
    ) -> list[Order]:
        if self.settings.execution_mode == "live":
            self.settings.assert_live_allowed()  # fails closed unless explicitly armed
        if self.kill_switch.engaged:
            raise KillSwitchEngaged(self.kill_switch.reason or "engaged")

        account = self.broker.get_account()
        if self.breaker.update(account.equity):
            self.kill_switch.engage("drawdown_circuit_breaker")
            self._audit(as_of, "PORTFOLIO", "halt", {"reason": "drawdown"})
            raise KillSwitchEngaged("drawdown circuit-breaker tripped")

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
        orders: list[Order] = []
        for symbol, weight in target_weights.items():
            price = prices.get(symbol)
            if not price or price <= 0:
                continue
            target_qty = weight * deployable / price
            delta = target_qty - positions.get(symbol, 0.0)
            if abs(delta) * price < 1.0:  # skip dust
                continue
            order = Order(
                client_order_id=f"{as_of.strftime('%Y%m%dT%H%M%S')}-{symbol}",
                symbol=symbol,
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                qty=abs(delta),
            )
            sanity_check_order(order, ref_price=price)
            if not self.rate_limiter.allow():
                raise OrderRateExceeded(f"order-rate cap {self.rate_limiter.max_orders} reached")
            filled = self.broker.submit_order(order)
            orders.append(filled)
            self._audit(
                as_of,
                symbol,
                "order",
                {
                    "client_order_id": order.client_order_id,
                    "side": order.side.value,
                    "qty": order.qty,
                    "status": filled.status.value,
                    "mode": self.settings.execution_mode,
                },
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

    def _audit(self, as_of: datetime, symbol: str, event: str, detail: dict) -> None:
        if self.audit_store is not None:
            log_execution_event(
                self.audit_store, as_of=as_of, symbol=symbol, event=event, detail=detail
            )
