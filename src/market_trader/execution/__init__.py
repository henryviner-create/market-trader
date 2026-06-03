"""Execution tier — the LAST stage, strictly downstream of risk.

PAPER-FIRST. Default mode is paper; live order routing is gated behind
``Settings.assert_live_allowed()`` (two switches + intent) and is **never enabled
autonomously**. Every order passes the guardrails (kill-switch, hard pre-trade
limits, drawdown breaker, sanity checks, order-rate cap, capital ceiling) before
it can reach a broker, and every action is audit-logged.
"""

from market_trader.execution.audit import (
    EXECUTION_DATASET,
    load_execution_audit,
    log_execution_event,
)
from market_trader.execution.broker import (
    Account,
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from market_trader.execution.engine import ExecutionEngine
from market_trader.execution.guardrails import (
    CapitalCeilingBreach,
    KillSwitch,
    KillSwitchEngaged,
    OrderRateExceeded,
    OrderRateLimiter,
    OrderSanityError,
    sanity_check_order,
)
from market_trader.execution.paper_broker import PaperBroker
from market_trader.execution.reconciliation import Divergence, reconcile

__all__ = [
    "EXECUTION_DATASET",
    "Account",
    "Broker",
    "CapitalCeilingBreach",
    "Divergence",
    "ExecutionEngine",
    "KillSwitch",
    "KillSwitchEngaged",
    "Order",
    "OrderRateExceeded",
    "OrderRateLimiter",
    "OrderSanityError",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperBroker",
    "Position",
    "load_execution_audit",
    "log_execution_event",
    "reconcile",
    "sanity_check_order",
]
