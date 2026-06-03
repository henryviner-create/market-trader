"""market-trader: a market intelligence & forecasting system.

Phase 0 establishes the foundations the rest of the system is measured against:

* a **bitemporal** canonical data model (event time vs. knowledge time),
* a knowledge-time clock and point-in-time data access,
* a purpose-built, leakage-resistant **validation / backtest harness**.

Nothing in this package may let a consumer observe a fact before its
``knowledge_time``. That single rule is what makes every later backtest honest.
"""

__version__ = "0.0.0"
